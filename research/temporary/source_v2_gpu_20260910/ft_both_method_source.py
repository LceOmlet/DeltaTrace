    @torch.no_grad()
    def calculate_ifr_multi_hop_both(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        n_hops: int = 1,
        renorm_threshold: Optional[float] = None,
        observation_mask: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> llm_attr.LLMAttributionResult:
        input_ids_all, attn_mask, prompt_len_full, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])

        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "multi_hop_both",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "stop_config": self._stop_config().__dict__,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        # Exclude EOS (assumed to be the last generation token).
        end_no_eos = int(gen_len) - 2
        if end_no_eos < 0:
            score_array = torch.full((int(gen_len), total_len), torch.nan, dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "multi_hop_both",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "stop_config": self._stop_config().__dict__,
                    "note": "No non-EOS generation tokens; returning NaN attribution matrix.",
                }
            }
            return self._finalize_result(score_array, metadata=metadata)

        # Scheme B: fill only sink_span rows; default to all_gen_span.
        if sink_span is None:
            span_start, span_end = (0, end_no_eos)
        else:
            span_start, span_end = sink_span
        if span_start < 0 or span_end < span_start or span_end > end_no_eos:
            raise ValueError(
                f"Invalid sink_span ({span_start}, {span_end}) for generation length {gen_len} with EOS excluded."
            )

        think_start_abs: Optional[int] = None
        think_end_abs: Optional[int] = None
        if thinking_span is not None:
            think_start, think_end = thinking_span
            if think_start < 0 or think_end < think_start or think_end > end_no_eos:
                raise ValueError(
                    f"Invalid thinking_span ({think_start}, {think_end}) for generation length {gen_len} with EOS excluded."
                )
            think_start_abs = int(prompt_len_full) + int(think_start)
            think_end_abs = int(prompt_len_full) + int(think_end)

        # Internal hop span: CoT+output when both spans are provided; otherwise default all_gen.
        if sink_span is not None and thinking_span is not None:
            all_gen_start = min(int(span_start), int(thinking_span[0]))
            all_gen_end = max(int(span_end), int(thinking_span[1]))
        else:
            all_gen_start = 0
            all_gen_end = int(end_no_eos)

        all_gen_start = max(0, int(all_gen_start))
        all_gen_end = min(int(end_no_eos), int(all_gen_end))
        if all_gen_end < all_gen_start:
            raise ValueError("Derived all_gen_span is empty after EOS exclusion and bounds checking.")

        sink_start_abs = int(prompt_len_full) + int(span_start)
        sink_end_abs = int(prompt_len_full) + int(span_end)
        all_gen_start_abs = int(prompt_len_full) + int(all_gen_start)
        all_gen_end_abs = int(prompt_len_full) + int(all_gen_end)

        obs_mask_tensor: Optional[torch.Tensor] = None
        if observation_mask is not None:
            obs_mask_tensor = torch.as_tensor(observation_mask, dtype=torch.float32)
            if obs_mask_tensor.ndim != 1:
                raise ValueError("observation_mask must be a 1D tensor or sequence.")
            if obs_mask_tensor.numel() == gen_len:
                mask_full = torch.zeros(total_len, dtype=torch.float32)
                mask_full[int(prompt_len_full) : int(prompt_len_full) + int(gen_len)] = obs_mask_tensor
                obs_mask_tensor = mask_full
            elif obs_mask_tensor.numel() != total_len:
                raise ValueError(
                    f"observation_mask must have length {gen_len} (generation) or {total_len} (full sequence)."
                )

        stop_keep_mask_full = _build_stop_keep_mask_full(
            prompt_len_full=int(prompt_len_full),
            gen_len=int(gen_len),
            user_prompt_indices=list(getattr(self, "user_prompt_indices", []) or []),
            user_prompt_tokens=list(getattr(self, "user_prompt_tokens", []) or []),
            generation_tokens=list(getattr(self, "generation_tokens", []) or []),
        )

        cache, attentions, metadata, weight_pack = self._capture_model_state(input_ids_all, attn_mask)
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        hop_count = max(0, int(n_hops))

        # Base hop: aggregate over all_gen_span, excluding stop tokens from the sink weights.
        gen_tokens = list(getattr(self, "generation_tokens", []) or [])
        span_stops = []
        for gi in range(int(all_gen_start), int(all_gen_end) + 1):
            tok = gen_tokens[gi] if 0 <= gi < len(gen_tokens) else ""
            span_stops.append(is_stop_token(tok))
        base_weights = None
        if any(span_stops):
            base_weights = torch.as_tensor(
                [0.0 if st else 1.0 for st in span_stops],
                dtype=params.model_dtype,
            )

        base_ifr_raw = compute_ifr_sentence_aggregate(
            sink_start=all_gen_start_abs,
            sink_end=all_gen_end_abs,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            sink_weights=base_weights,
        )
        base_total = torch.nan_to_num(
            base_ifr_raw.token_importance_total.to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        base_total = base_total * stop_keep_mask_full
        base_ifr = IFRAggregate(
            per_layer=base_ifr_raw.per_layer,
            token_importance_total=base_total,
            head_importance_total=base_ifr_raw.head_importance_total,
            ffn_importance_per_layer=base_ifr_raw.ffn_importance_per_layer,
            resid_ffn_importance_per_layer=base_ifr_raw.resid_ffn_importance_per_layer,
        )

        # Observation mask: respect provided observation_mask, otherwise follow in_all_gen scheme B,
        # and then apply stop-token masking so stop tokens are removed from the observation path.
        if obs_mask_tensor is None:
            obs_mask = torch.ones((total_len,), dtype=torch.float32)
            obs_mask[int(all_gen_start_abs) : min(int(all_gen_end_abs) + 1, total_len)] = 0.0
            if int(all_gen_end_abs) + 1 < total_len:
                obs_mask[int(all_gen_end_abs) + 1 :] = 0.0
        else:
            obs_mask = obs_mask_tensor.clone().to(dtype=torch.float32)
            if int(obs_mask.shape[0]) != int(total_len):
                raise ValueError("observation_mask must match sequence length.")

        obs_mask = obs_mask * stop_keep_mask_full

        base_obs = base_total.clone() * obs_mask
        obs_accum = base_obs.clone()
        per_hop_obs: List[torch.Tensor] = []

        raw_attributions: List[IFRAggregate] = [base_ifr]

        denom_base = float(base_total.sum().item())
        w_span_raw = base_total[int(all_gen_start_abs) : int(all_gen_end_abs) + 1].detach().clone()
        w_span_raw = torch.nan_to_num(w_span_raw, nan=0.0, posinf=0.0, neginf=0.0)
        w_span_sum = float(w_span_raw.sum().item())
        w_span_weights = (
            (w_span_raw / (w_span_sum + 1e-12))
            if w_span_sum > 0.0
            else torch.zeros_like(w_span_raw, dtype=torch.float32)
        )
        current_ratio = float(w_span_sum) / (denom_base + 1e-12) if denom_base > 0.0 else 0.0
        ratios: List[float] = [float(current_ratio)]

        for hop in range(1, hop_count + 1):
            if float(w_span_sum) <= 0.0 or float(current_ratio) <= 0.0:
                zeros = torch.zeros_like(base_total)
                for _ in range(hop, hop_count + 1):
                    raw_attributions.append(
                        IFRAggregate(
                            per_layer=[],
                            token_importance_total=zeros,
                            head_importance_total=torch.zeros_like(base_ifr.head_importance_total),
                            ffn_importance_per_layer=torch.zeros_like(base_ifr.ffn_importance_per_layer),
                            resid_ffn_importance_per_layer=torch.zeros_like(base_ifr.resid_ffn_importance_per_layer),
                        )
                    )
                    per_hop_obs.append(torch.zeros_like(base_total))
                    ratios.append(0.0)
                break

            hop_ifr_raw = compute_ifr_sentence_aggregate(
                sink_start=all_gen_start_abs,
                sink_end=all_gen_end_abs,
                cache=cache,
                attentions=attentions,
                weight_pack=weight_pack,
                params=params,
                renorm_threshold=renorm,
                sink_weights=w_span_weights,
            )

            hop_total = torch.nan_to_num(
                hop_ifr_raw.token_importance_total.to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            hop_total = hop_total * stop_keep_mask_full
            hop_ifr = IFRAggregate(
                per_layer=hop_ifr_raw.per_layer,
                token_importance_total=hop_total,
                head_importance_total=hop_ifr_raw.head_importance_total,
                ffn_importance_per_layer=hop_ifr_raw.ffn_importance_per_layer,
                resid_ffn_importance_per_layer=hop_ifr_raw.resid_ffn_importance_per_layer,
            )
            raw_attributions.append(hop_ifr)

            obs_only = hop_total * obs_mask * float(current_ratio)
            obs_accum += obs_only
            per_hop_obs.append(obs_only)

            w_span_raw = hop_total[int(all_gen_start_abs) : int(all_gen_end_abs) + 1].detach().clone()
            w_span_raw = torch.nan_to_num(w_span_raw, nan=0.0, posinf=0.0, neginf=0.0)
            w_span_sum = float(w_span_raw.sum().item())
            w_span_weights = (
                (w_span_raw / (w_span_sum + 1e-12))
                if w_span_sum > 0.0
                else torch.zeros_like(w_span_raw, dtype=torch.float32)
            )

            hop_denom = float(hop_total.sum().item())
            if hop_denom <= 0.0:
                current_ratio = 0.0
            else:
                current_ratio *= float(w_span_sum) / (hop_denom + 1e-12)
            ratios.append(float(current_ratio))

        obs_avg = obs_accum / float(max(1, hop_count))
        observation: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "mask": obs_mask,
            "base": base_obs,
            "per_hop": per_hop_obs,
            "sum": obs_accum,
            "avg": obs_avg,
        }

        multi_hop = MultiHopIFRResult(raw_attributions=raw_attributions, thinking_ratios=ratios, observation=observation)
        eval_vector = multi_hop.observation["sum"]

        score_array = torch.full((int(gen_len), total_len), torch.nan, dtype=torch.float32)
        for offset in range(int(span_start), int(span_end) + 1):
            tok = gen_tokens[offset] if 0 <= offset < len(gen_tokens) else ""
            if is_stop_token(tok):
                continue
            score_array[offset] = eval_vector

        projected_per_hop = [self._project_vector(result.token_importance_total) for result in multi_hop.raw_attributions]
        obs = multi_hop.observation
        observation_projected = {
            "mask": self.extract_user_prompt_attributions(self.prompt_tokens, obs["mask"].view(1, -1))[0],
            "base": self._project_vector(obs["base"]),
            "sum": self._project_vector(obs["sum"]),
            "avg": self._project_vector(obs["avg"]),
            "per_hop": [self._project_vector(vec) for vec in obs["per_hop"]],
        }

        meta: Dict[str, Any] = {
            "ifr": {
                "type": "multi_hop_both",
                "sink_span_generation": (int(span_start), int(span_end)),
                "sink_span_absolute": (int(sink_start_abs), int(sink_end_abs)),
                "thinking_span_generation": (
                    (int(thinking_span[0]), int(thinking_span[1])) if thinking_span is not None else None
                ),
                "thinking_span_absolute": (
                    (int(think_start_abs), int(think_end_abs)) if think_start_abs is not None else None
                ),
                "all_gen_span_generation": (int(all_gen_start), int(all_gen_end)),
                "all_gen_span_absolute": (int(all_gen_start_abs), int(all_gen_end_abs)),
                "renorm_threshold": float(renorm),
                "n_hops": int(n_hops),
                "thinking_ratios": multi_hop.thinking_ratios,
                "per_hop_projected": projected_per_hop,
                "observation_projected": observation_projected,
                "stop_config": self._stop_config().__dict__,
                "raw": multi_hop,
                "note": "both = stop_words + in_all_gen (scheme B; hops over all_gen_span with EOS excluded).",
            }
        }

        return self._finalize_result(score_array, metadata=meta)
