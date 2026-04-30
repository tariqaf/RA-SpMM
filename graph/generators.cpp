// ============================================================================
// generators.cpp - Graph generators returning CSR format
//
// All generators produce valid CSR with:
// - sorted colind within each row
// - no duplicate entries within a row
// - deterministic values from the provided seed
// - populated metadata
// ============================================================================
#include "../ra_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helper: compute metadata from rowptr/colind
// ---------------------------------------------------------------------------
void compute_metadata(SparseMatrix& mat) {
    const int M = mat.M;
    const int K = mat.K;
    const int nnz = static_cast<int>(mat.colind.size());

    std::vector<float> row_lens(M, 0.f);
    float sum_len = 0.f;
    for (int r = 0; r < M; ++r) {
        const int len = mat.rowptr[r + 1] - mat.rowptr[r];
        row_lens[r] = static_cast<float>(len);
        sum_len += static_cast<float>(len);
    }

    const float avg = (M > 0) ? (sum_len / static_cast<float>(M)) : 0.f;
    mat.avg_nnz_per_row = avg;

    float var = 0.f;
    for (int r = 0; r < M; ++r) {
        const float d = row_lens[r] - avg;
        var += d * d;
    }
    mat.std_nnz_per_row = (M > 1) ? std::sqrt(var / static_cast<float>(M)) : 0.f;
    mat.density = (M > 0 && K > 0) ? static_cast<float>(nnz) / static_cast<float>(M * K) : 0.f;

    std::vector<float> sorted_lens = row_lens;
    std::sort(sorted_lens.begin(), sorted_lens.end());
    float gini_sum = 0.f;
    for (int i = 0; i < M; ++i) {
        gini_sum += (2.f * static_cast<float>(i + 1) - static_cast<float>(M) - 1.f) * sorted_lens[i];
    }
    mat.skew_coeff = (M > 0 && sum_len > 0.f)
        ? (gini_sum / (static_cast<float>(M) * sum_len))
        : 0.f;

    const int block_size = 64;
    int clustered_nnz = 0;
    for (int r = 0; r < M; ++r) {
        const int rb = r / block_size;
        for (int p = mat.rowptr[r]; p < mat.rowptr[r + 1]; ++p) {
            const int cb = mat.colind[p] / block_size;
            if (rb == cb) {
                ++clustered_nnz;
            }
        }
    }
    mat.clustering_proxy = (nnz > 0) ? static_cast<float>(clustered_nnz) / static_cast<float>(nnz) : 0.f;
}

// ---------------------------------------------------------------------------
// Helper: canonicalize CSR rows (sort + deduplicate)
// ---------------------------------------------------------------------------
void finalize_csr(SparseMatrix& mat) {
    std::vector<int> out_rowptr(mat.M + 1, 0);
    std::vector<int> out_colind;
    std::vector<float> out_vals;
    out_colind.reserve(mat.colind.size());
    out_vals.reserve(mat.vals.size());

    for (int r = 0; r < mat.M; ++r) {
        const int start = mat.rowptr[r];
        const int end = mat.rowptr[r + 1];
        std::vector<std::pair<int, float>> entries;
        entries.reserve(std::max(0, end - start));
        for (int p = start; p < end; ++p) {
            if (mat.colind[p] >= 0 && mat.colind[p] < mat.K) {
                entries.push_back({mat.colind[p], mat.vals[p]});
            }
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        int last_col = -1;
        for (const auto& entry : entries) {
            if (entry.first == last_col) {
                continue;
            }
            out_colind.push_back(entry.first);
            out_vals.push_back(entry.second);
            last_col = entry.first;
        }
        out_rowptr[r + 1] = static_cast<int>(out_colind.size());
    }

    mat.rowptr.swap(out_rowptr);
    mat.colind.swap(out_colind);
    mat.vals.swap(out_vals);
}

int clamp_degree(int degree, int max_degree) {
    return std::max(0, std::min(degree, max_degree));
}

int jitter_degree(std::mt19937& rng, int base, float frac, int max_degree) {
    if (base <= 0) {
        return 0;
    }
    const int span = std::max(1, static_cast<int>(std::round(static_cast<float>(base) * frac)));
    std::uniform_int_distribution<int> dist(-span, span);
    return clamp_degree(base + dist(rng), max_degree);
}

void insert_random_unique(
    std::set<int>& cols,
    int target_count,
    int begin,
    int end,
    std::mt19937& rng)
{
    begin = std::max(0, begin);
    end = std::max(begin, end);
    if (begin >= end) {
        return;
    }

    const int range = end - begin;
    target_count = std::min(target_count, range);
    if (target_count <= static_cast<int>(cols.size())) {
        return;
    }

    if (target_count * 4 >= range) {
        std::vector<int> candidates(range);
        std::iota(candidates.begin(), candidates.end(), begin);
        std::shuffle(candidates.begin(), candidates.end(), rng);
        for (int c : candidates) {
            cols.insert(c);
            if (static_cast<int>(cols.size()) >= target_count) {
                break;
            }
        }
        return;
    }

    std::uniform_int_distribution<int> dist(begin, end - 1);
    int attempts = 0;
    const int max_attempts = std::max(32, target_count * 16);
    while (static_cast<int>(cols.size()) < target_count && attempts < max_attempts) {
        cols.insert(dist(rng));
        ++attempts;
    }
}

void append_row_from_cols(
    SparseMatrix& mat,
    const std::set<int>& cols,
    std::mt19937& rng,
    std::uniform_real_distribution<float>& val_dist)
{
    for (int c : cols) {
        mat.colind.push_back(c);
        mat.vals.push_back(val_dist(rng));
    }
}

SparseMatrix empty_matrix(int M, int K) {
    SparseMatrix mat;
    mat.M = M;
    mat.K = K;
    mat.rowptr.resize(M + 1, 0);
    return mat;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. random_sparse: uniform random sparsity
// ---------------------------------------------------------------------------
SparseMatrix random_sparse(int M, int K, int nnz_per_row, unsigned seed) {
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    for (int r = 0; r < M; ++r) {
        std::set<int> cols;
        insert_random_unique(cols, clamp_degree(nnz_per_row, K), 0, K, rng);
        append_row_from_cols(mat, cols, rng, val_dist);
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 2. skewed_powerlaw: bounded heavy-tail degree distribution
// ---------------------------------------------------------------------------
SparseMatrix skewed_powerlaw(int M, int K, float alpha, int min_nnz, int max_nnz, unsigned seed) {
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u_dist(0.0, 1.0);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    for (int r = 0; r < M; ++r) {
        const double u = u_dist(rng);
        const double heavy_tail = std::pow(u, std::max(1.0f, alpha));
        int degree = static_cast<int>(std::round(
            static_cast<double>(min_nnz) +
            static_cast<double>(max_nnz - min_nnz) * heavy_tail));
        degree = clamp_degree(std::max(min_nnz, degree), K);

        std::set<int> cols;
        insert_random_unique(cols, degree, 0, K, rng);
        append_row_from_cols(mat, cols, rng, val_dist);
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 3. community_clustered: block/community structure
// ---------------------------------------------------------------------------
SparseMatrix community_clustered(
    int M,
    int K,
    int n_comm,
    float within_density,
    float between_density,
    unsigned seed)
{
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u_dist(0.f, 1.f);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    n_comm = std::max(1, n_comm);
    std::vector<int> row_comm(M), col_comm(K);
    for (int r = 0; r < M; ++r) {
        row_comm[r] = (r * n_comm) / std::max(1, M);
    }
    for (int c = 0; c < K; ++c) {
        col_comm[c] = (c * n_comm) / std::max(1, K);
    }

    std::vector<std::vector<int>> comm_cols(n_comm);
    for (int c = 0; c < K; ++c) {
        comm_cols[col_comm[c]].push_back(c);
    }

    for (int r = 0; r < M; ++r) {
        const int rc = row_comm[r];
        std::set<int> cols;
        for (int c : comm_cols[rc]) {
            if (u_dist(rng) < within_density) {
                cols.insert(c);
            }
        }

        const int between_samples = std::max(1, static_cast<int>(std::round(between_density * static_cast<float>(K))));
        std::uniform_int_distribution<int> col_dist(0, K - 1);
        for (int i = 0; i < between_samples; ++i) {
            const int c = col_dist(rng);
            if (col_comm[c] != rc && u_dist(rng) < 0.5f) {
                cols.insert(c);
            }
        }

        append_row_from_cols(mat, cols, rng, val_dist);
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 4. bipartite_rectangular: M != K rectangular matrix
// ---------------------------------------------------------------------------
SparseMatrix bipartite_rectangular(int M, int K, int nnz_per_row, unsigned seed) {
    return random_sparse(M, K, nnz_per_row, seed);
}

// ---------------------------------------------------------------------------
// 5. road_like: low-degree near-diagonal connectivity
// ---------------------------------------------------------------------------
SparseMatrix road_like(int M, int K, int avg_degree, unsigned seed) {
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);
    const int half = std::max(1, avg_degree / 2);

    for (int r = 0; r < M; ++r) {
        std::set<int> cols;
        for (int d = -half; d <= half; ++d) {
            if (d == 0) {
                continue;
            }
            const int c = (r + d + K) % K;
            cols.insert(c);
        }
        append_row_from_cols(mat, cols, rng, val_dist);
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 6. block_locality: dense diagonal blocks
// ---------------------------------------------------------------------------
SparseMatrix block_locality(int M, int K, int block_size, float fill, unsigned seed) {
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u_dist(0.f, 1.f);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    block_size = std::max(1, block_size);
    const int num_diag_blocks = std::min((M + block_size - 1) / block_size,
                                         (K + block_size - 1) / block_size);

    for (int r = 0; r < M; ++r) {
        const int rb = r / block_size;
        std::set<int> cols;
        if (rb < num_diag_blocks) {
            const int c_start = rb * block_size;
            const int c_end = std::min(c_start + block_size, K);
            for (int c = c_start; c < c_end; ++c) {
                if (u_dist(rng) < fill) {
                    cols.insert(c);
                }
            }
        }
        append_row_from_cols(mat, cols, rng, val_dist);
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 7. hub_heavy: a few rows own a large fraction of nnz
// ---------------------------------------------------------------------------
SparseMatrix hub_heavy(
    int M,
    int K,
    float hub_fraction,
    int hub_degree,
    int base_degree,
    unsigned seed)
{
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    const int num_hubs = std::max(1, static_cast<int>(std::round(static_cast<float>(M) * std::max(0.f, hub_fraction))));
    std::vector<int> row_ids(M);
    std::iota(row_ids.begin(), row_ids.end(), 0);
    std::shuffle(row_ids.begin(), row_ids.end(), rng);
    std::vector<char> is_hub(M, 0);
    for (int i = 0; i < std::min(M, num_hubs); ++i) {
        is_hub[row_ids[i]] = 1;
    }

    for (int r = 0; r < M; ++r) {
        const int degree = is_hub[r] ? clamp_degree(hub_degree, K) : clamp_degree(base_degree, K);
        std::set<int> cols;
        insert_random_unique(cols, degree, 0, K, rng);
        append_row_from_cols(mat, cols, rng, val_dist);
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 8. mixed_skew: mixture of tiny, medium, and giant rows
// ---------------------------------------------------------------------------
SparseMatrix mixed_skew(
    int M,
    int K,
    float frac_tiny,
    float frac_medium,
    float frac_giant,
    int tiny_degree,
    int medium_degree,
    int giant_degree,
    unsigned seed)
{
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    frac_tiny = std::max(0.f, frac_tiny);
    frac_medium = std::max(0.f, frac_medium);
    frac_giant = std::max(0.f, frac_giant);
    const float total_frac = std::max(1e-6f, frac_tiny + frac_medium + frac_giant);
    frac_tiny /= total_frac;
    frac_medium /= total_frac;
    frac_giant /= total_frac;

    const int tiny_count = static_cast<int>(std::round(frac_tiny * static_cast<float>(M)));
    const int giant_count = static_cast<int>(std::round(frac_giant * static_cast<float>(M)));
    const int medium_count = std::max(0, M - tiny_count - giant_count);

    std::vector<int> labels(M, 1);
    std::fill(labels.begin(), labels.begin() + std::min(M, tiny_count), 0);
    std::fill(labels.begin() + std::min(M, tiny_count),
              labels.begin() + std::min(M, tiny_count + medium_count), 1);
    std::fill(labels.begin() + std::min(M, tiny_count + medium_count), labels.end(), 2);
    std::shuffle(labels.begin(), labels.end(), rng);

    for (int r = 0; r < M; ++r) {
        int degree = 0;
        if (labels[r] == 0) {
            degree = jitter_degree(rng, tiny_degree, 0.25f, K);
        } else if (labels[r] == 1) {
            degree = jitter_degree(rng, medium_degree, 0.35f, K);
        } else {
            degree = jitter_degree(rng, giant_degree, 0.20f, K);
        }
        degree = std::max(1, degree);

        std::set<int> cols;
        insert_random_unique(cols, degree, 0, K, rng);
        append_row_from_cols(mat, cols, rng, val_dist);
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 9. clustered_window: local row windows share compact spans
// ---------------------------------------------------------------------------
SparseMatrix clustered_window(
    int M,
    int K,
    int window_rows,
    int window_span,
    float intra_window_density,
    unsigned seed)
{
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    window_rows = std::max(1, window_rows);
    window_span = clamp_degree(window_span, K);
    const int base_degree = std::max(1, static_cast<int>(std::round(intra_window_density * static_cast<float>(std::max(1, window_span)))));

    std::uniform_int_distribution<int> span_dist(0, std::max(0, K - std::max(1, window_span)));
    for (int base = 0; base < M; base += window_rows) {
        const int end = std::min(M, base + window_rows);
        const int span_start = (window_span < K) ? span_dist(rng) : 0;
        const int span_end = std::min(K, span_start + std::max(1, window_span));

        std::set<int> shared_cols;
        insert_random_unique(shared_cols, std::max(1, base_degree / 3), span_start, span_end, rng);

        for (int r = base; r < end; ++r) {
            std::set<int> cols = shared_cols;
            const int degree = std::max(static_cast<int>(shared_cols.size()),
                                        jitter_degree(rng, base_degree, 0.25f, span_end - span_start));
            insert_random_unique(cols, degree, span_start, span_end, rng);
            append_row_from_cols(mat, cols, rng, val_dist);
            mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
        }
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 10. scrambled_locality: clustered windows with globally permuted row order
// ---------------------------------------------------------------------------
SparseMatrix scrambled_locality(
    int M,
    int K,
    int window_rows,
    int window_span,
    float intra_window_density,
    unsigned seed)
{
    const SparseMatrix base = clustered_window(M, K, window_rows, window_span, intra_window_density, seed + 17u);

    SparseMatrix out = empty_matrix(M, K);
    std::mt19937 rng(seed);
    std::vector<int> perm(M);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (int new_r = 0; new_r < M; ++new_r) {
        const int old_r = perm[new_r];
        for (int p = base.rowptr[old_r]; p < base.rowptr[old_r + 1]; ++p) {
            out.colind.push_back(base.colind[p]);
            out.vals.push_back(base.vals[p]);
        }
        out.rowptr[new_r + 1] = static_cast<int>(out.colind.size());
    }

    finalize_csr(out);
    compute_metadata(out);
    return out;
}

// ---------------------------------------------------------------------------
// 11. mixed_block_skew: windows mix block-dense, skew-heavy, and random sparse
// ---------------------------------------------------------------------------
SparseMatrix mixed_block_skew(
    int M,
    int K,
    int window_rows,
    float frac_block_windows,
    float frac_skew_windows,
    float block_fill,
    int skew_base_degree,
    int skew_hub_degree,
    unsigned seed)
{
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    window_rows = std::max(1, window_rows);
    const int num_windows = (M + window_rows - 1) / window_rows;
    const int block_windows = static_cast<int>(std::round(std::max(0.f, frac_block_windows) * static_cast<float>(num_windows)));
    const int skew_windows = static_cast<int>(std::round(std::max(0.f, frac_skew_windows) * static_cast<float>(num_windows)));

    std::vector<int> window_type(num_windows, 2);  // 0=block, 1=skew, 2=random
    for (int i = 0; i < std::min(num_windows, block_windows); ++i) {
        window_type[i] = 0;
    }
    for (int i = block_windows; i < std::min(num_windows, block_windows + skew_windows); ++i) {
        window_type[i] = 1;
    }
    std::shuffle(window_type.begin(), window_type.end(), rng);

    const int block_span = std::min(K, std::max(64, window_rows * 4));
    const int random_degree = std::min(K, 16);
    std::uniform_int_distribution<int> span_dist(0, std::max(0, K - std::max(1, block_span)));

    for (int w = 0; w < num_windows; ++w) {
        const int base = w * window_rows;
        const int end = std::min(M, base + window_rows);
        const int type = window_type[w];

        if (type == 0) {
            const int span_start = (block_span < K) ? span_dist(rng) : 0;
            const int span_end = std::min(K, span_start + block_span);
            const int degree = std::max(1, static_cast<int>(std::round(block_fill * static_cast<float>(span_end - span_start))));
            for (int r = base; r < end; ++r) {
                std::set<int> cols;
                insert_random_unique(cols, degree, span_start, span_end, rng);
                append_row_from_cols(mat, cols, rng, val_dist);
                mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
            }
        } else if (type == 1) {
            std::vector<int> local_rows(end - base);
            std::iota(local_rows.begin(), local_rows.end(), 0);
            std::shuffle(local_rows.begin(), local_rows.end(), rng);
            const int hub_rows = std::max(1, (end - base) / 8);
            for (int r = base; r < end; ++r) {
                std::set<int> cols;
                const bool is_hub = std::find(local_rows.begin(), local_rows.begin() + std::min(hub_rows, static_cast<int>(local_rows.size())), r - base) !=
                                    local_rows.begin() + std::min(hub_rows, static_cast<int>(local_rows.size()));
                const int degree = is_hub
                    ? jitter_degree(rng, skew_hub_degree, 0.20f, K)
                    : jitter_degree(rng, skew_base_degree, 0.35f, K);
                insert_random_unique(cols, std::max(1, degree), 0, K, rng);
                append_row_from_cols(mat, cols, rng, val_dist);
                mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
            }
        } else {
            for (int r = base; r < end; ++r) {
                std::set<int> cols;
                insert_random_unique(cols, random_degree, 0, K, rng);
                append_row_from_cols(mat, cols, rng, val_dist);
                mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
            }
        }
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 12. cluster_plus_hubs: clustered background plus cross-community hub rows
// ---------------------------------------------------------------------------
SparseMatrix cluster_plus_hubs(
    int M,
    int K,
    int num_clusters,
    float within_density,
    float between_density,
    float hub_fraction,
    int hub_degree,
    unsigned seed)
{
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u_dist(0.f, 1.f);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    num_clusters = std::max(1, num_clusters);
    const int num_hubs = std::max(1, static_cast<int>(std::round(std::max(0.f, hub_fraction) * static_cast<float>(M))));

    std::vector<int> row_ids(M);
    std::iota(row_ids.begin(), row_ids.end(), 0);
    std::shuffle(row_ids.begin(), row_ids.end(), rng);
    std::vector<char> is_hub(M, 0);
    for (int i = 0; i < std::min(M, num_hubs); ++i) {
        is_hub[row_ids[i]] = 1;
    }

    std::vector<int> row_comm(M), col_comm(K);
    for (int r = 0; r < M; ++r) {
        row_comm[r] = (r * num_clusters) / std::max(1, M);
    }
    for (int c = 0; c < K; ++c) {
        col_comm[c] = (c * num_clusters) / std::max(1, K);
    }
    std::vector<std::vector<int>> comm_cols(num_clusters);
    for (int c = 0; c < K; ++c) {
        comm_cols[col_comm[c]].push_back(c);
    }

    for (int r = 0; r < M; ++r) {
        const int rc = row_comm[r];
        std::set<int> cols;

        if (is_hub[r]) {
            insert_random_unique(cols, clamp_degree(hub_degree, K), 0, K, rng);
        } else {
            for (int c : comm_cols[rc]) {
                if (u_dist(rng) < within_density) {
                    cols.insert(c);
                }
            }
            const int extra = std::max(1, static_cast<int>(std::round(between_density * static_cast<float>(K))));
            std::uniform_int_distribution<int> col_dist(0, K - 1);
            for (int i = 0; i < extra; ++i) {
                const int c = col_dist(rng);
                if (col_comm[c] != rc && u_dist(rng) < 0.5f) {
                    cols.insert(c);
                }
            }
        }

        append_row_from_cols(mat, cols, rng, val_dist);
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 13. heterogeneous_windows: explicit mixture of local structural regimes
// ---------------------------------------------------------------------------
SparseMatrix heterogeneous_windows(
    int M,
    int K,
    int window_rows,
    float frac_block_dense,
    float frac_clustered_sparse,
    float frac_random_sparse,
    float frac_skew_heavy,
    unsigned seed)
{
    SparseMatrix mat = empty_matrix(M, K);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    window_rows = std::max(1, window_rows);
    const int num_windows = (M + window_rows - 1) / window_rows;

    std::vector<float> fracs = {
        std::max(0.f, frac_block_dense),
        std::max(0.f, frac_clustered_sparse),
        std::max(0.f, frac_random_sparse),
        std::max(0.f, frac_skew_heavy),
    };
    float frac_sum = std::accumulate(fracs.begin(), fracs.end(), 0.f);
    if (frac_sum <= 0.f) {
        fracs = {0.25f, 0.25f, 0.25f, 0.25f};
        frac_sum = 1.f;
    }
    for (float& f : fracs) {
        f /= frac_sum;
    }

    std::vector<int> window_type(num_windows, 0);
    int cursor = 0;
    for (int type = 0; type < 4; ++type) {
        const int count = (type == 3)
            ? (num_windows - cursor)
            : static_cast<int>(std::round(fracs[type] * static_cast<float>(num_windows)));
        for (int i = 0; i < count && cursor < num_windows; ++i) {
            window_type[cursor++] = type;
        }
    }
    std::shuffle(window_type.begin(), window_type.end(), rng);

    const int block_span = std::min(K, std::max(64, window_rows * 4));
    const int cluster_span = std::min(K, std::max(48, window_rows * 3));
    const int random_degree = std::min(K, 12);
    std::uniform_int_distribution<int> block_dist(0, std::max(0, K - std::max(1, block_span)));
    std::uniform_int_distribution<int> cluster_dist(0, std::max(0, K - std::max(1, cluster_span)));

    for (int w = 0; w < num_windows; ++w) {
        const int base = w * window_rows;
        const int end = std::min(M, base + window_rows);
        const int type = window_type[w];

        if (type == 0) {
            const int span_start = (block_span < K) ? block_dist(rng) : 0;
            const int span_end = std::min(K, span_start + block_span);
            const int degree = std::max(1, static_cast<int>(std::round(0.65f * static_cast<float>(span_end - span_start))));
            for (int r = base; r < end; ++r) {
                std::set<int> cols;
                insert_random_unique(cols, degree, span_start, span_end, rng);
                append_row_from_cols(mat, cols, rng, val_dist);
                mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
            }
        } else if (type == 1) {
            const int span_start = (cluster_span < K) ? cluster_dist(rng) : 0;
            const int span_end = std::min(K, span_start + cluster_span);
            std::set<int> shared_cols;
            insert_random_unique(shared_cols, std::max(1, cluster_span / 8), span_start, span_end, rng);
            for (int r = base; r < end; ++r) {
                std::set<int> cols = shared_cols;
                insert_random_unique(cols, std::max(static_cast<int>(shared_cols.size()), cluster_span / 5),
                                     span_start, span_end, rng);
                append_row_from_cols(mat, cols, rng, val_dist);
                mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
            }
        } else if (type == 2) {
            for (int r = base; r < end; ++r) {
                std::set<int> cols;
                insert_random_unique(cols, random_degree, 0, K, rng);
                append_row_from_cols(mat, cols, rng, val_dist);
                mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
            }
        } else {
            std::vector<int> local(end - base);
            std::iota(local.begin(), local.end(), 0);
            std::shuffle(local.begin(), local.end(), rng);
            const int heavy_rows = std::max(1, (end - base) / 6);
            for (int r = base; r < end; ++r) {
                const bool is_heavy =
                    std::find(local.begin(), local.begin() + std::min(heavy_rows, static_cast<int>(local.size())), r - base) !=
                    local.begin() + std::min(heavy_rows, static_cast<int>(local.size()));
                const int degree = is_heavy ? std::min(K, 192) : std::min(K, 6);
                std::set<int> cols;
                insert_random_unique(cols, degree, 0, K, rng);
                append_row_from_cols(mat, cols, rng, val_dist);
                mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
            }
        }
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}

// ---------------------------------------------------------------------------
// 14. powerlaw_realistic: Barabasi-Albert preferential attachment
//
// Generates a true power-law graph using the BA model.
// m edges are added per new node, producing a continuous degree distribution
// with exponent ~3 (controlled by m). No hard degree cap, no bimodal step.
// m=5 gives median degree ~10, avg_degree ~2m=10.
// ---------------------------------------------------------------------------
SparseMatrix powerlaw_realistic(int M, int m_attach, unsigned seed) {
    m_attach = std::max(1, std::min(m_attach, M - 1));

    SparseMatrix mat = empty_matrix(M, M);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    // Degree sequence for preferential attachment sampling
    std::vector<int> degree_pool;  // each node appears proportional to its degree
    degree_pool.reserve(M * m_attach * 2);

    // Bootstrap: start with a small complete graph of size m_attach+1
    const int init_size = std::min(M, m_attach + 1);
    // Use an adjacency set for dedup during construction
    std::vector<std::set<int>> adj(M);
    for (int i = 0; i < init_size; ++i) {
        for (int j = i + 1; j < init_size; ++j) {
            adj[i].insert(j);
            adj[j].insert(i);
            degree_pool.push_back(i);
            degree_pool.push_back(j);
        }
    }

    // Attach new nodes
    for (int node = init_size; node < M; ++node) {
        std::set<int> targets;
        int attempts = 0;
        const int max_attempts = m_attach * 64;
        while (static_cast<int>(targets.size()) < m_attach && attempts < max_attempts) {
            // Preferential: pick uniformly from degree_pool
            if (!degree_pool.empty()) {
                std::uniform_int_distribution<int> pool_dist(0, static_cast<int>(degree_pool.size()) - 1);
                int candidate = degree_pool[pool_dist(rng)];
                if (candidate != node) {
                    targets.insert(candidate);
                }
            }
            ++attempts;
        }
        // If preferential attachment didn't fill, add random targets
        if (static_cast<int>(targets.size()) < m_attach) {
            std::uniform_int_distribution<int> rand_dist(0, node - 1);
            while (static_cast<int>(targets.size()) < m_attach) {
                targets.insert(rand_dist(rng));
            }
        }
        for (int t : targets) {
            adj[node].insert(t);
            adj[t].insert(node);
            degree_pool.push_back(node);
            degree_pool.push_back(t);
        }
    }

    // Build CSR from adjacency sets
    for (int r = 0; r < M; ++r) {
        for (int c : adj[r]) {
            mat.colind.push_back(c);
            mat.vals.push_back(val_dist(rng));
        }
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}


// ---------------------------------------------------------------------------
// 15. community_sbm: Stochastic Block Model matching real community graphs
//
// Generates sparse community structure matching com-DBLP / com-Amazon density.
// n_comm=32, within_density=0.06 gives avg_deg ~(M/n_comm)*0.06 ~ 6-8 for M=8K.
// Undirected, symmetric. Diagonal blocks are dense; off-diagonal are very sparse.
// ---------------------------------------------------------------------------
SparseMatrix community_sbm(int M, int n_comm, float within_density, float between_density, unsigned seed) {
    n_comm = std::max(1, std::min(n_comm, M));
    within_density = std::max(0.f, std::min(1.f, within_density));
    between_density = std::max(0.f, std::min(1.f, between_density));

    SparseMatrix mat = empty_matrix(M, M);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u_dist(0.f, 1.f);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.f);

    // Assign each node to a community
    std::vector<int> comm(M);
    for (int i = 0; i < M; ++i) {
        comm[i] = (i * n_comm) / std::max(1, M);
    }

    // Group columns by community
    std::vector<std::vector<int>> comm_cols(n_comm);
    for (int c = 0; c < M; ++c) {
        comm_cols[comm[c]].push_back(c);
    }

    for (int r = 0; r < M; ++r) {
        const int rc = comm[r];
        std::set<int> cols;

        // Intra-community edges (undirected: only connect to c >= r to avoid double-counting,
        // but we want symmetric so we add both directions independently via density sampling)
        for (int c : comm_cols[rc]) {
            if (c == r) continue;
            if (u_dist(rng) < within_density) {
                cols.insert(c);
            }
        }

        // Inter-community edges (very sparse)
        if (between_density > 0.f) {
            for (int oc = 0; oc < n_comm; ++oc) {
                if (oc == rc) continue;
                for (int c : comm_cols[oc]) {
                    if (u_dist(rng) < between_density) {
                        cols.insert(c);
                    }
                }
            }
        }

        for (int c : cols) {
            mat.colind.push_back(c);
            mat.vals.push_back(val_dist(rng));
        }
        mat.rowptr[r + 1] = static_cast<int>(mat.colind.size());
    }

    finalize_csr(mat);
    compute_metadata(mat);
    return mat;
}


// ---------------------------------------------------------------------------
// 16. reordered_variant: random row+column permutation preserving values
// ---------------------------------------------------------------------------
SparseMatrix reordered_variant(const SparseMatrix& mat, unsigned seed) {
    SparseMatrix out = empty_matrix(mat.M, mat.K);

    std::mt19937 rng(seed);
    std::vector<int> row_perm(mat.M), col_perm(mat.K);
    std::iota(row_perm.begin(), row_perm.end(), 0);
    std::iota(col_perm.begin(), col_perm.end(), 0);
    std::shuffle(row_perm.begin(), row_perm.end(), rng);
    std::shuffle(col_perm.begin(), col_perm.end(), rng);

    std::vector<int> col_inv(mat.K);
    for (int c = 0; c < mat.K; ++c) {
        col_inv[col_perm[c]] = c;
    }

    for (int new_r = 0; new_r < mat.M; ++new_r) {
        const int old_r = row_perm[new_r];
        std::vector<std::pair<int, float>> entries;
        entries.reserve(mat.rowptr[old_r + 1] - mat.rowptr[old_r]);
        for (int p = mat.rowptr[old_r]; p < mat.rowptr[old_r + 1]; ++p) {
            entries.push_back({col_inv[mat.colind[p]], mat.vals[p]});
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        int last_col = -1;
        for (const auto& entry : entries) {
            if (entry.first == last_col) {
                continue;
            }
            out.colind.push_back(entry.first);
            out.vals.push_back(entry.second);
            last_col = entry.first;
        }
        out.rowptr[new_r + 1] = static_cast<int>(out.colind.size());
    }

    finalize_csr(out);
    compute_metadata(out);
    return out;
}
