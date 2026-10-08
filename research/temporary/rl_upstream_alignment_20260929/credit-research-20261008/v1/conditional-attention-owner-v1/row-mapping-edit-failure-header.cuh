// Research-only row statistics for the existing finite-FA owner.
// No model forward, attention parser, mask, scheduler or backward is replaced.
// Excluding the largest key has its own normalization scale. This avoids
// subtracting a rounded unit probability to recover a small remaining mass.
struct ConditionalTopRow {
    float top, top_key, top_u;
    float other_max, other_sum, other_u;

    __device__ ConditionalTopRow()
        : top(-INFINITY), top_key(INFINITY), top_u(0.f),
          other_max(-INFINITY), other_sum(0.f), other_u(0.f) {}

    __device__ void add_other(float score, float uv) {
        if (score == -INFINITY) return;
        const float next=fmaxf(other_max,score);
        const float left=other_sum==0.f?0.f:expf(other_max-next);
        const float right=expf(score-next);
        other_sum=other_sum*left+right;
        other_u=other_u*left+right*uv;
        other_max=next;
    }

    __device__ void add(float score, int key, float uv) {
        if (score>top || (score==top && float(key)<top_key)) {
            add_other(top,top_u);
            top=score;top_key=float(key);top_u=uv;
        } else add_other(score,uv);
    }

    __device__ void finish(float &logz,float &logz_other,float &arg,
                           float &mean,float &mean_other) {
        // Match this pinned MACA owner's quad_allreduce_ row-lane mapping.\n        // Its Allreduce<64> combines lanes xor48/32/16; CUDA xor2/1 mixes rows.\n        flash::MaxOp<float> max_op;
        flash::SumOp<float> sum_op;
        const float largest=flash::Allreduce<64>::run(top,max_op);
        const float selected=flash::Allreduce<64>::run(
            top==largest?-top_key:-INFINITY,max_op);
        arg=-selected;
        const float selected_u=flash::Allreduce<64>::run(
            top_key==arg?top_u:0.f,sum_op);
        if (top_key!=arg) add_other(top,top_u);
        const float shared_max=flash::Allreduce<64>::run(other_max,max_op);
        const float factor=other_sum==0.f?0.f:expf(other_max-shared_max);
        const float mass=flash::Allreduce<64>::run(other_sum*factor,sum_op);
        const float weighted=flash::Allreduce<64>::run(other_u*factor,sum_op);
        logz_other=mass==0.f?-INFINITY:shared_max+logf(mass);
        mean_other=mass==0.f?0.f:weighted/mass;
        const float high=fmaxf(largest,logz_other);
        logz=high+logf(expf(largest-high)+expf(logz_other-high));
        mean=expf(largest-logz)*selected_u+expf(logz_other-logz)*mean_other;
    }
};

__device__ inline float conditional_logaddexp(float x,float y) {
    const float high=fmaxf(x,y);
    return high==-INFINITY?-INFINITY:high+logf(expf(x-high)+expf(y-high));
}

// Stable divided difference of sigmoid. The derivative limit is part of
// the formula; there is no threshold, clipping or attribution rescaling.
__device__ inline float conditional_sigmoid(float x) {
    const float e=expf(-fabsf(x));
    return x>=0.f?1.f/(1.f+e):e/(1.f+e);
}

__device__ inline float conditional_sigmoid_secant(float x,float y) {
    const float distance=fabsf(x-y);
    const float numerator=conditional_sigmoid(fmaxf(x,y))*
                          conditional_sigmoid(-fminf(x,y));
    return numerator*(distance==0.f?1.f:-expm1f(-distance)/distance);
}

template<typename Params>
__device__ inline void conditional_exclusive_row(const Params &p,int64_t pos,
    int key,float score,float uv,float &logz_other,float &mean_other) {
    if (float(key)==p.conditional_top_key[pos]) {
        logz_other=p.conditional_excluded_lse[pos];
        mean_other=p.conditional_excluded_u[pos];
    } else {
        const float probability=expf(score-p.conditional_factual_lse[pos]);
        // A non-largest key has probability at most 1/2. The top-key case
        // above never uses 1-p or a subtraction to recover its missing mass.
        logz_other=p.conditional_factual_lse[pos]+log1pf(-probability);
        mean_other=(p.conditional_factual_u[pos]-probability*uv)/(1.f-probability);
    }
}
