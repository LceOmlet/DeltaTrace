"""SQL model-template boundary; trajectory behavior belongs to SkyRL."""


def configure_sql_tokenizer(tokenizer, template_path=None):
    if template_path:
        from pathlib import Path
        # Pinned SkyRL's original template leaves <think> in the actual action,
        # where its original format reward expects it.
        tokenizer.chat_template = Path(template_path).read_text()


def make_sql_environments(configuration, tokenizer):
    # Existing entry name delegates to the original owner loop as well.
    from sql_owner_rollout import make_sql_owner_environments
    return make_sql_owner_environments(configuration, tokenizer)
