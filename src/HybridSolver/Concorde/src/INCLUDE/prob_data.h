#ifndef HYBRIDCONCORDE_PROB_DATA_H
#define HYBRIDCONCORDE_PROB_DATA_H


#define TRACE() fprintf(stderr, "%s (%d): %s\n", __FILE__, __LINE__, __func__)
#define FRAC_TOL (0.000001)

#define NODE_SEL(lb, prob, best_lb, best_prob) \
    (((lb) < (best_lb) || ((lb == best_lb ) && (prob) > (best_prob))) ? 1 : 0)


#define H_TSP_BRANCH_STRONG_VAL(v0,v1,w)                 \
    (((v0) < (v1) ? ((w) * (v0) + (v1))                  \
                  : ((w) * (v1) + (v0)))                 \
                    / (w + 1.0))                    


typedef struct NNedges_probs{
    int                 ecount;
    int                 ncount;
    int                 total_pnode_ties; // total number of times we had a tie in the lower bound of an open node for node selection
    int                 pnode_ties_count; // used times where where there was a tie in the lower bound of an open node and the one with highest optimality score was selected
    int                 total_pvar_ties;  // total number of times we had a tie in the strong branching value of an edge for variable selection
    int                 pvar_ties_count;  // used times where there was a tie in the strong branching value of an edge and the one with highest prob was selected
    int                 total_frac_ties; // total number of times we had a tie in the fractional value of an edge for variable selection
    int                 frac_ties_count; // used times where there was a tie in the fractional value of an edge and the one with highest prob was selected
    double              **edge_probs;
    char                filename[64] ;
} NNedges_probs;

extern NNedges_probs nn_edges_probs;


#endif //HYBRIDCONCORDE_PROB_DATA_H
