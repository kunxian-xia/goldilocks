#ifndef NTT_GOLDILOCKS
#define NTT_GOLDILOCKS

#include "goldilocks_base_field.hpp"
#include <cassert>
#include <gmp.h>
#include <omp.h>

#define NUM_PHASES 3
#define NUM_BLOCKS 1

class NTT_Goldilocks
{
private:
    u_int32_t s = 0;
    u_int32_t nThreads;
    uint64_t nqr;
    // N = maxDomainSize
    // w = `N`-th root of unity
    // roots = { w^0, w^1, w^2, ..., w^(N - 1) }
    // powTwoInv = { 1, 2^(-1), ..., 2^(-s) } and it's used to get N^(-1) which we need in iFFT
    Goldilocks::Element *roots; 
    Goldilocks::Element *powTwoInv;
    // r = shift
    // r = { 1, r^1, r^2, ..., r^}
    // r_ = N^(-1) * r
    Goldilocks::Element *r;
    Goldilocks::Element *r_;
    int extension;
    bool enable_avx = false;

    static u_int32_t log2(u_int64_t size)
    {
        {
            assert(size != 0);
            u_int32_t res = 0;
            while (size != 1)
            {
                size >>= 1;
                res++;
            }
            return res;
        }
    }
    void NTT_iters(
        Goldilocks::Element *dst,
        Goldilocks::Element *src,
        u_int64_t size,
        u_int64_t offset_cols,
        u_int64_t ncols,
        u_int64_t ncols_all,
        u_int64_t nphase,
        Goldilocks::Element *aux,
        bool inverse,
        bool extend);

    inline int intt_idx(int i, int N)
    {
        int ind1 = N - i;
        if (ind1 == N)
        {
            ind1 = 0;
        }
        return ind1;
    }

public:
    // compute the following data for NTT
    // 1. s (2-adicity)
    // 2. roots = { 1, w^1, w^2, ..., w^(N-1) } where w is the $N$-th root of unity (N = maxDomainSize)
    // 3. powTwoInv = { 1, 2^(-1), 2^(-2), ..., 2^(-s) }
    NTT_Goldilocks(u_int64_t maxDomainSize, u_int32_t _nThreads = 0, int extension_ = 1, bool enable_avx_ = false)
    {
        enable_avx = enable_avx_;
        r = NULL;
        r_ = NULL;
        if (maxDomainSize == 0)
            return;
        nThreads = _nThreads == 0 ? omp_get_max_threads() : _nThreads;
        extension = extension_;

        // n = domainPow, N = maxDomainSize
        // N = 2^n
        u_int32_t domainPow = NTT_Goldilocks::log2(maxDomainSize);
        assert(1UL << domainPow == maxDomainSize);

        mpz_t m_qm1d2;
        mpz_t m_q;
        mpz_t m_nqr;
        mpz_t m_aux;
        mpz_init(m_qm1d2);
        mpz_init(m_q);
        mpz_init(m_nqr);
        mpz_init(m_aux);

        u_int64_t negone = GOLDILOCKS_PRIME - 1;

        mpz_import(m_aux, 1, 1, sizeof(u_int64_t), 0, 0, &negone);
        // m_q = goldilocks_prime
        mpz_add_ui(m_q, m_aux, 1); 
        // (m_q-1)/2
        mpz_fdiv_q_2exp(m_qm1d2, m_aux, 1); 

        // note(kunxian): 
        //   use the loop to find a quadratic non residue (nqr)
        //   modulo the goldilocks_prime
        mpz_set_ui(m_nqr, 2); 
        // start to try with m_nqr = 2
        // m_aux = 2^((q-1)/2) (mod q)
        mpz_powm(m_aux, m_nqr, m_qm1d2, m_q); 
        while (mpz_cmp_ui(m_aux, 1) == 0) 
        {
            mpz_add_ui(m_nqr, m_nqr, 1);
            mpz_powm(m_aux, m_nqr, m_qm1d2, m_q);
        }

        // 2-adicity of poseidon is 32
        // s = domainPow
        // q - 1 = 2^s * 2^(32 - s) * t
        s = 1;
        mpz_set(m_aux, m_qm1d2); // m_aux = (q-1) / 2
        while ((!mpz_tstbit(m_aux, 0)) && (s < domainPow))
        {
            // m_aux = (m_aux-1)/2
            mpz_fdiv_q_2exp(m_aux, m_aux, 1);
            s++;
        }

        nqr = mpz_get_ui(m_nqr);

        if (s < domainPow)
        {
            throw std::range_error("Domain size too big for the curve");
        }
        // s <= 32
        assert(s <= 32);
        uint64_t nRoots = 1LL << s;

        // roots = { 1, w^1, w^2, ...., w^(N-1) } where w is `maxDomainSize`-th root of unity
        roots = (Goldilocks::Element *)malloc(nRoots * sizeof(Goldilocks::Element));
        // powTwoInv = { 1, 2^(-1), 2^(-2), ..., 2^(-s) }
        powTwoInv = (Goldilocks::Element *)malloc((s + 1) * sizeof(Goldilocks::Element));

        roots[0] = Goldilocks::one();
        powTwoInv[0] = Goldilocks::one();

        if (nRoots > 1)
        {
            // calculate the first root of unity
            // mpz_powm(m_aux, m_nqr, m_aux, m_q);
            // roots[1] = Goldilocks::fromU64(mpz_get_ui(m_aux));
            roots[1] = Goldilocks::w(domainPow);

            // modular inverse of 2
            mpz_set_ui(m_aux, 2);
            mpz_invert(m_aux, m_aux, m_q);
            // gmp_printf("2^(-1) = %Zd\n", m_aux);
            // 2^(-1) = 2^63 - 2^31 + 1
            // 2*(2^63 - 2^31 + 1) = 2^64 - 2^32 + 2 = p + 1
            powTwoInv[1] = Goldilocks::fromU64(mpz_get_ui(m_aux));
        }

        // calculate the rest of roots of unity
        for (uint64_t i = 2; i < nRoots; i++)
        {
            // roots[i] = w^i
            roots[i] = roots[i - 1] * roots[1];
        }
        // aux = w^N = 1
        Goldilocks::Element aux = roots[nRoots - 1] * roots[1];
        assert(Goldilocks::toU64(aux) == 1);

        for (uint64_t i = 2; i <= s; i++)
        {
            powTwoInv[i] = powTwoInv[i - 1] * powTwoInv[1];
        }

        mpz_clear(m_qm1d2);
        mpz_clear(m_q);
        mpz_clear(m_nqr);
        mpz_clear(m_aux);
    };

    ~NTT_Goldilocks()
    {
        if (s != 0)
        {
            free(roots);
            free(powTwoInv);
        }
        if (r != NULL)
        {
            delete (r);
        }
        if (r_ != NULL)
        {
            delete (r_);
        }
    }
    inline void computeR(int N)
    {
        // r[i] = shift^i
        // r = { 1, shift^1, shift^2, ..., shift^(N-1) }
        // r_ = N^(-1) * r 
        u_int64_t domainPow = log2(N);
        r = new Goldilocks::Element[N];
        r_ = new Goldilocks::Element[N];
        r[0] = Goldilocks::one();
        // r_[0] = N^(-1)
        r_[0] = powTwoInv[domainPow];
        for (int i = 1; i < N; i++)
        {
            // note(kunxian): used to gen coset = shift * unity_of_roots group.
            Goldilocks::mul(r[i], r[i - 1], Goldilocks::shift());
            Goldilocks::mul(r_[i], r[i], powTwoInv[domainPow]);
        }
    }
    // doc:
    // run ntt on src, and ntt result is written to dst
    // handles `ncols` polynomials 
    // len(src) = size * cols * sizeof(Goldilocks::Element)
    void NTT(Goldilocks::Element *dst,
             Goldilocks::Element *src,
             u_int64_t size,
             u_int64_t ncols = 1,
             Goldilocks::Element *buffer = NULL,
             u_int64_t nphase = NUM_PHASES, // TODO: nphase ?? 
             u_int64_t nblock = NUM_BLOCKS, // TODO: nblock ??
             bool inverse = false,
             bool extend = false);
    void INTT(Goldilocks::Element *dst,
              Goldilocks::Element *src,
              u_int64_t size,
              u_int64_t ncols = 1,
              Goldilocks::Element *buffer = NULL,
              u_int64_t nphase = NUM_PHASES,
              u_int64_t nblock = NUM_BLOCKS,
              bool extend = false);
    void reversePermutation(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all);

    // return the twiddle factor w'^idx at layer domainPow.
    inline Goldilocks::Element &root(u_int32_t domainPow, u_int64_t idx)
    {
        // w' = w^(s-domain)
        // w'^(domainPow) = 1 => w' is the `domainPow`-th root of unity
        // w'^(idx)
        return roots[idx << (s - domainPow)];
    }
    // input has length N * ncols
    // output has length N_Extended * ncols
    void extendPol(Goldilocks::Element *output,
                   Goldilocks::Element *input,
                   uint64_t N_Extended,
                   uint64_t N,
                   uint64_t ncols,
                   Goldilocks::Element *buffer = NULL,
                   u_int64_t nphase = NUM_PHASES,
                   u_int64_t nblock = NUM_BLOCKS);
};

#endif
