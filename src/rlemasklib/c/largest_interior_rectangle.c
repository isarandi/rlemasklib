#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "basics.h"
#include "minmax.h"
#include "pad_crop.h"
#include "shapes.h"
#include "boolfuncs.h"
#include "largest_interior_rectangle.h"

struct Stack {
    // we grow this stack backwards
    // when empty, it points to stack_base (one beyond the allocation)
    // pushing is *(--pStack)=elem, popping is elem=*(pStack++)
    uint *start;
    uint *end;
    uint *p;
};
static void stackInit(struct Stack *stack, siz n) ;
static void stackFree(struct Stack *stack);
static inline void stackPush(struct Stack *stack, uint elem);
static inline uint stackPop(struct Stack *stack);
static inline void stackClear(struct Stack *stack);
static inline bool stackIsEmpty(struct Stack *stack);
static inline uint stackPeek(struct Stack *stack);

static void rleLeftContours(const RLE *R, RLE *M);
static void rleRightContours(const RLE *R, RLE *M);
static void rleLeftRightContours(const RLE *R, RLE *R_left, RLE *R_right);
static uint uintAbsDiff(uint a, uint b);

struct BestRect {
    uint area;
    uint right;
    uint width;
    uint top;
};
static inline void updateIfBetter(
    uint sx, struct Stack *stack, uint x, uint y, struct BestRect *bestRect);

struct BestRectAspect {
    double height;
    double width;
    double right;
    double top;
};
static inline void updateIfBetterAspect(
    uint sx, struct Stack *stack, uint x, uint y, struct BestRectAspect *bestRect, double aspect_ratio);

void rleLargestInteriorRectangle(const RLE *R_, uint* rect_out) {
    // If no foreground, return zeros
    if (R_->m <= 1 || R_->h == 0 || R_->w == 0) {
        rect_out[0] = rect_out[1] = rect_out[2] = rect_out[3] = 0;
        return;
    }

    // Crop the RLE to its bounding box so we don't waste memory and time on empty regions
    uint bbox[4];
    rleToUintBbox(R_, bbox);
    RLE R_cropped;
    RLE* R;
    // Crop only if there's significant amount of empty space to crop
    if (R_->w > bbox[2] + 20 || R_->h > bbox[3] + 20) {
        rleCrop(R_, &R_cropped, 1, bbox);
        R = &R_cropped;
    } else {
        bbox[0] = 0;
        bbox[1] = 0;
        R = R_;
    }
    siz h = R->h;
    siz w = R->w;

    // Get the left and right contours (the pixels where foreground starts or ends in a row)
    // That is, these are foreground pixels that have a background pixel to their left or right
    RLE R_left;
    RLE R_right;
    rleLeftRightContours(R, &R_left, &R_right);
    if (R == &R_cropped) {
        rleFree(&R_cropped);
    }
    uint *cnts_left = R_left.cnts;
    uint cnt_left = cnts_left[0];
    siz i_cnt_left = 0;
    uint *cnts_right = R_right.cnts;
    uint cnt_right = cnts_right[0];
    siz i_cnt_right = 0;

    // start_x is the x position where the currently active horizontal run started
    // if a certain row has no active run, it is UINT_MAX
    uint *start_x = malloc(h * sizeof(uint));
    for (siz i = 0; i < h; i++) {
        start_x[i] = UINT_MAX;
    }

    // the stack contains the row indices where there was an expansion of width
    struct Stack stack;
    stackInit(&stack, h);

    // to keep track of the best rectangle found so far
    struct BestRect bestRect;
    bestRect.area = 0;

    for (siz x = 0; x < w; x++) {
        // Set the start map of the pixels that start the foreground
        // the value is the current x
        siz y = 0;
        while (true) {
            uint yend = y + cnt_left;
            uint yend_real = yend < h ? yend : h;
            if (i_cnt_left % 2 == 1) {
                while (y < yend_real) {
                    start_x[y++] = x;
                }
            } else {
                y = yend_real;
            }
            if (yend < h) {
                i_cnt_left++;
                cnt_left = cnts_left[i_cnt_left]; // we surely aren't at the end of the RLE
            } else if (yend > h) {
                cnt_left = yend - h;
                break;
            } else {
                i_cnt_left++;
                if (i_cnt_left < R_left.m) {  // we may be at the very end of the RLE
                    cnt_left = cnts_left[i_cnt_left];
                }
                break;
            }
        }

        // any_finals == Are there any pixels in this col that end the foreground?
        // i.e. foreground pixels where their right neighbor is background (or the end of the row)
        // we check this by looking at the forward-difference RLE and whether we
        // are in a foreground run of it or whether the current background run ends within this col
        // if no pixel ends the foreground, there is no sense in checking for maximal rectangles
        // since any rectangle we'd find would be trivially extendable to the right.
        bool any_finals = i_cnt_right % 2 == 1 || cnt_right < h;
        if (any_finals) {
            stackClear(&stack);
            uint start_cur = start_x[0];
            for (y = 1; y < h; y++) {
                uint start_next = start_x[y];

                if (start_cur == start_next) {
                    // noop
                } else if (start_cur > start_next) {
                    stackPush(&stack, y - 1); // push the row index to the stack
                } else {
                    updateIfBetter(start_cur, &stack, x, y, &bestRect);
                    while (!stackIsEmpty(&stack) && start_x[stackPeek(&stack)] < start_next) {
                        start_cur = start_x[stackPop(&stack)];
                        updateIfBetter(start_cur, &stack, x, y, &bestRect);
                    }
                }
                start_cur = start_next;
            }
            updateIfBetter(start_cur, &stack, x, h, &bestRect);

            while (!stackIsEmpty(&stack)) {
                start_cur = start_x[stackPop(&stack)];
                updateIfBetter(start_cur, &stack, x, h, &bestRect);
            }
        }

        // Set the start map to UINT_MAX for the pixels that end the foreground
        // Even if there are no pixels that end the foreground, we still need to move to the next column
        // so we have to churn through the cnt_right values.
        //
        // But this is only needed if there will be more columns to process
        // if the loop ends here, then it doesn't matter
        y = 0;
        while (true) {
            uint yend = y + cnt_right;
            uint yend_real = yend < h ? yend : h;
            if (i_cnt_right % 2 == 1) {
                while (y < yend_real) {
                    start_x[y++] = UINT_MAX;
                }
            } else {
                y = yend_real;
            }
            if (yend < h) {
                i_cnt_right++;
                cnt_right = cnts_right[i_cnt_right];
            } else if (yend > h) {
                cnt_right = yend - h;
                break;
            } else {
                i_cnt_right++;
                if (i_cnt_right < R_right.m) {
                    cnt_right = cnts_right[i_cnt_right];
                }
                break;
            }
        }
    }
    free(start_x);
    stackFree(&stack);
    rleFree(&R_left);
    rleFree(&R_right);

    // add the bbox offset in case we cropped at the start and copy the result to the output
    rect_out[0] = bestRect.right + 1 - bestRect.width + bbox[0];
    rect_out[1] = bestRect.top + bbox[1];
    rect_out[2] = bestRect.width;
    rect_out[3] = bestRect.area / bestRect.width;
}

static inline void updateIfBetter(
    uint sx, struct Stack *stack, uint x, uint y, struct BestRect *bestRect) {
    if (sx == UINT_MAX) {
        return;
    }

    uint top = stackIsEmpty(stack) ? 0 : stackPeek(stack) + 1;
    uint bw = x + 1 - sx;
    uint bh = y - top;
    uint area = bw * bh;
    if (area > bestRect->area) {
        bestRect->area = area;
        bestRect->right = x;
        bestRect->width = bw;
        bestRect->top = top;
    }
}


void rleLargestInteriorRectangleAspect(const RLE *R_, double* rect_out, double aspect_ratio) {
    // If no foreground, return zeros
    if (R_->m <= 1 || R_->h == 0 || R_->w == 0) {
        rect_out[0] = rect_out[1] = rect_out[2] = rect_out[3] = 0;
        return;
    }

    // Crop the RLE to its bounding box so we don't waste memory and time on empty regions
    uint bbox[4];
    rleToUintBbox(R_, bbox);
    RLE R_cropped;
    RLE* R;
    // Crop only if there's significant amount of empty space to crop
    if (R_->w > bbox[2] + 20 || R_->h > bbox[3] + 20) {
        rleCrop(R_, &R_cropped, 1, bbox);
        R = &R_cropped;
    } else {
        bbox[0] = 0;
        bbox[1] = 0;
        R = R_;
    }
    siz h = R->h;
    siz w = R->w;

    // Get the left and right contours (the pixels where foreground starts or ends in a row)
    // That is, these are foreground pixels that have a background pixel to their left or right
    RLE R_left;
    RLE R_right;
    rleLeftRightContours(R, &R_left, &R_right);
    if (R == &R_cropped) {
        rleFree(&R_cropped);
    }
    uint *cnts_left = R_left.cnts;
    uint cnt_left = cnts_left[0];
    siz i_cnt_left = 0;
    uint *cnts_right = R_right.cnts;
    uint cnt_right = cnts_right[0];
    siz i_cnt_right = 0;

    // start_x is the x position where the currently active horizontal run started
    // if a certain row has no active run, it is UINT_MAX
    uint *start_x = malloc(h * sizeof(uint));
    for (siz i = 0; i < h; i++) {
        start_x[i] = UINT_MAX;
    }

    // the stack contains the row indices where there was an expansion of width
    struct Stack stack;
    stackInit(&stack, h);

    // to keep track of the best rectangle found so far
    struct BestRectAspect bestRect;
    bestRect.height = 0;
    bestRect.width = 0;

    for (siz x = 0; x < w; x++) {
        // Set the start map of the pixels that start the foreground
        // the value is the current x
        siz y = 0;
        while (true) {
            uint yend = y + cnt_left;
            uint yend_real = yend < h ? yend : h;
            if (i_cnt_left % 2 == 1) {
                while (y < yend_real) {
                    start_x[y++] = x;
                }
            } else {
                y = yend_real;
            }
            if (yend < h) {
                i_cnt_left++;
                cnt_left = cnts_left[i_cnt_left]; // we surely aren't at the end of the RLE
            } else if (yend > h) {
                cnt_left = yend - h;
                break;
            } else {
                i_cnt_left++;
                if (i_cnt_left < R_left.m) {  // we may be at the very end of the RLE
                    cnt_left = cnts_left[i_cnt_left];
                }
                break;
            }
        }

        // any_finals == Are there any pixels in this col that end the foreground?
        // i.e. foreground pixels where their right neighbor is background (or the end of the row)
        // we check this by looking at the forward-difference RLE and whether we
        // are in a foreground run of it or whether the current background run ends within this col
        // if no pixel ends the foreground, there is no sense in checking for maximal rectangles
        // since any rectangle we'd find would be trivially extendable to the right.
        bool any_finals = i_cnt_right % 2 == 1 || cnt_right < h;
        if (any_finals) {
            stackClear(&stack);
            uint start_cur = start_x[0];
            for (y = 1; y < h; y++) {
                uint start_next = start_x[y];

                if (start_cur == start_next) {
                    // noop
                } else if (start_cur > start_next) {
                    stackPush(&stack, y - 1); // push the row index to the stack
                } else {
                    updateIfBetterAspect(start_cur, &stack, x, y, &bestRect, aspect_ratio);
                    while (!stackIsEmpty(&stack) && start_x[stackPeek(&stack)] < start_next) {
                        start_cur = start_x[stackPop(&stack)];
                        updateIfBetterAspect(start_cur, &stack, x, y, &bestRect, aspect_ratio);
                    }
                }
                start_cur = start_next;
            }
            updateIfBetterAspect(start_cur, &stack, x, h, &bestRect, aspect_ratio);

            while (!stackIsEmpty(&stack)) {
                start_cur = start_x[stackPop(&stack)];
                updateIfBetterAspect(start_cur, &stack, x, h, &bestRect, aspect_ratio);
            }
        }

        // Set the start map to UINT_MAX for the pixels that end the foreground
        // Even if there are no pixels that end the foreground, we still need to move to the next column
        // so we have to churn through the cnt_right values.
        //
        // But this is only needed if there will be more columns to process
        // if the loop ends here, then it doesn't matter
        y = 0;
        while (true) {
            uint yend = y + cnt_right;
            uint yend_real = yend < h ? yend : h;
            if (i_cnt_right % 2 == 1) {
                while (y < yend_real) {
                    start_x[y++] = UINT_MAX;
                }
            } else {
                y = yend_real;
            }
            if (yend < h) {
                i_cnt_right++;
                cnt_right = cnts_right[i_cnt_right];
            } else if (yend > h) {
                cnt_right = yend - h;
                break;
            } else {
                i_cnt_right++;
                if (i_cnt_right < R_right.m) {
                    cnt_right = cnts_right[i_cnt_right];
                }
                break;
            }
        }
    }
    free(start_x);
    stackFree(&stack);
    rleFree(&R_left);
    rleFree(&R_right);

    // add the bbox offset in case we cropped at the start and copy the result to the output
    rect_out[0] = bestRect.right + 1 - bestRect.width + bbox[0];
    rect_out[1] = bestRect.top + bbox[1];
    rect_out[2] = bestRect.width;
    rect_out[3] = bestRect.height;
}

static inline void updateIfBetterAspect(
    uint sx, struct Stack *stack, uint x, uint y, struct BestRectAspect *bestRect,
    double aspect_ratio) {

    if (sx == UINT_MAX) {
        return;
    }

    uint top = stackIsEmpty(stack) ? 0 : stackPeek(stack) + 1;
    uint bw = x + 1 - sx;
    uint bh = y - top;

    double current_ratio = (double)bw / (double)bh;
    if (current_ratio > aspect_ratio) {
        if (bh > bestRect->height) {
            double new_width = aspect_ratio * bh;
            bestRect->right = x - (bw - new_width) / 2;
            bestRect->width = new_width;
            bestRect->height = bh;
            bestRect->top = top;
        }
    } else {
        if (bw > bestRect->width) {
            double new_height = bw / aspect_ratio;
            bestRect->right = x;
            bestRect->width = bw;
            bestRect->height = new_height;
            bestRect->top = top + (bh - new_height) / 2;
        }
    }
}


static void rleLeftRightContours(const RLE *R, RLE *R_left, RLE *R_right) {
    if (R->m <= 1 || R->h == 0 || R->w == 0) {
        rleZeros(R_left, R->h, R->w);
        rleZeros(R_right, R->h, R->w);
        return;
    }

    if (R->w == 1) {
        rleCopy(R, R_left);
        rleCopy(R, R_right);
        return;
    }

    RLE shifted;
    rleCopy(R, &shifted);
    shifted.cnts[0] += shifted.h;  // expand the first run of 0s

    // remove last column (by changing the size of the last run and setting the num of runs)
    // this way the rest of the run info is not lost, so we don't have to make a new copy
    uint r = 0;
    for (siz i = shifted.m - 1; i > 0; i--) {
        r += shifted.cnts[i];
        if (r > R->h) {
            shifted.cnts[i] = r - R->h;
            shifted.m = i + 1;
            break;
        }
    }
    rleMerge2(R, &shifted, R_left, BOOLFUNC_SUB);
    // undo the changes to shifted, restoring shifted to the original state of R
    shifted.cnts[0] = R->cnts[0];
    shifted.cnts[shifted.m - 1] = R->cnts[shifted.m - 1];
    shifted.m = R->m;

    // remove first column
    r = 0;
    for (siz i = 0; i < shifted.m; i++) {
        r += shifted.cnts[i];
        if (r > R->h) {
            shifted.cnts[i] = r - R->h;
            if (i % 2 == 0) {
                // this is a zeros run (i.e. the top pixel of the second column is 0)
                // we can just set the cnts pointer to this run and adjust the num of runs
                shifted.cnts += i;
                shifted.m -= i;
            } else {
                // this is a ones run, we need to set the previous run of 0s to length 0,
                // and point cnts to that run of 0s (the RLE has to start with a run of 0s)
                // and then adjust the num of runs.
                shifted.cnts[i-1] = 0;
                shifted.cnts += i - 1;
                shifted.m -= i - 1;
            }
            break;
        }
    }
    // expand last run of 0s with a new column, or add a new run of 0s of size `height`
    if (shifted.m % 2 == 0) {
        // last run is 1s, we make space for a new run
        rleRealloc(&shifted, shifted.m + 1);
        shifted.cnts[shifted.m - 1] = shifted.h;
    } else {
        // last run is 0s, we can just enlarge it
        shifted.cnts[shifted.m - 1] += shifted.h;
    }
    rleMerge2(R, &shifted, R_right, BOOLFUNC_SUB);
    rleFree(&shifted);
}

void rleLargestInteriorRectangleAroundCenter(
    const RLE *R, double* rect_out, uint cy, uint cx, double aspect_ratio) {
    // If there are no foreground pixels, return zeros
    if (R->m <= 1 || R->h == 0 || R->w == 0) {
        rect_out[0] = rect_out[1] = rect_out[2] = rect_out[3] = 0;
        return;
    }
    // If the center is in the background, return zeros
    if (rleGet(R, cy, cx) == 0) {
        rect_out[0] = rect_out[1] = rect_out[2] = rect_out[3] = 0;
        return;
    }

    uint bbox[4];
    rleToUintBbox(R, bbox);
    siz histsize = uintMin(cx - bbox[0] + 1, bbox[0] + bbox[2] - cx);
    uint *hist = malloc(histsize * sizeof(uint));
    uint max_hist = uintMin(cy - bbox[1], bbox[1] + bbox[3] - 1 - cy);
    for (siz i = 0; i < histsize; i++) {
        hist[i] = max_hist;
    }
    uint xdist_earliest_zero = histsize;

    uint r = 0;
    uint next_target = (cx - histsize + 1) * R->h + cy;
    for (siz j=0; j < R->m; j++) {
        uint cnt = R->cnts[j];
        uint r_end = r + cnt;
        if (r_end > next_target) { // this is the run that contains the next pixel in row cy
            uint xstart = r / R->h;
            uint xlastcol = (r_end - 1) / R->h;
            if (xstart == xlastcol) { // the run is within a single column
                uint xdist = uintAbsDiff(cx, xstart);
                if (xdist < xdist_earliest_zero) {
                    if (j % 2 == 0) { // run of zeros
                        xdist_earliest_zero = xdist;
                    } else {
                        uint ystart = r % R->h;
                        uint ylast = ystart + cnt - 1;
                        uint value = uintMin(cy - ystart, ylast - cy);
                        if (value < hist[xdist]) {
                            hist[xdist] = value;
                        }
                    }
                }
                next_target += R->h;
            } else {
                // this many pixels of row cy are in this run
                uint runwidth = (r_end - next_target) / R->h + 1;
                if (j % 2 == 0) { // run of zeros
                    // its enough to set the smallest distance to 0
                    // the smallest distance depends on whether the run is to left or right of cx
                    // cx cannot be part of a 0s run -- we checked that at the start
                    uint xfirst_cy = next_target / R->h;
                    uint xlast_cy = (r_end - cy) / R->h;
                    uint xdist = xlast_cy < cx ? cx - xlast_cy : xfirst_cy - cx;
                    if (xdist < xdist_earliest_zero) {
                        xdist_earliest_zero = xdist;
                    }
                } else { // run of 1s
                    uint ystart = r % R->h;
                    if (ystart <= cy) {  // ystart might be after cy
                        // first column -- it only has a top limit at ystart
                        uint xdist = uintAbsDiff(cx, xstart);
                        if (xdist < xdist_earliest_zero) {
                            uint value = cy - ystart;
                            if (value < hist[xdist]) {
                                hist[xdist] = value;
                            }
                        }
                    }
                    uint ylast = (r_end - 1) % R->h;
                    if (ylast >= cy) {  // ylast might be before cy
                        // last column -- it only has a bottom limit at ylast
                        uint xdist = uintAbsDiff(cx, xlastcol);
                        if (xdist < xdist_earliest_zero) {
                            uint value = ylast - cy;
                            if (value < hist[xdist]) {
                                hist[xdist] = value;
                            }
                        }
                    }
                }
                next_target += R->h * runwidth; // the serial position of the next y=cy pixel
            }
            if (next_target / R->h >= cx + xdist_earliest_zero) {
                // if we have reached the last column where a nonzero column can be found, we stop
                break;
            }
        }
        r = r_end;
    }

    // the histogram is now complete
    // We now have to find the largest rectangle in it starting at the beginning.
    double best_area = 0;
    double best_hist = 0;
    uint min_hist = max_hist;
    for (siz i = 0; i < xdist_earliest_zero; i++) {
        if (hist[i] == hist[i+1]) {
            continue;
        }
        if (hist[i] < min_hist) {
            min_hist = hist[i];
        }
        uint height = min_hist * 2 + 1;
        uint width = i * 2 + 1;

        if (aspect_ratio == 0){
            uint area = height * width;
            if (area > best_area) {
                best_area = area;
                best_hist = min_hist;
            }
        } else {
            double current_ratio = (double)width / (double)height;
            if (current_ratio > aspect_ratio) {
                double new_width = aspect_ratio * height;
                double area = new_width * height;
                if (area > best_area) {
                    best_area = area;
                    best_hist = min_hist;
                }
            } else {
                double new_height = width / aspect_ratio;
                double area = width * new_height;
                if (area > best_area) {
                    best_area = area;
                    best_hist = min_hist - (height - new_height) / 2;
                }
            }
        }
    }

    free(hist);
    double best_height = best_hist * 2 + 1;
    double best_width = best_area / best_height;
    double best_x = cx - (best_width - 1) / 2;
    double best_y = cy - best_hist;
    rect_out[0] = best_x;
    rect_out[1] = best_y;
    rect_out[2] = best_width;
    rect_out[3] = best_height;
}

static uint uintAbsDiff(uint a, uint b) {
    return a > b ? a - b : b - a;
}


static void stackInit(struct Stack *stack, siz n) {
    stack->start = malloc(n * sizeof(uint));
    stack->end = stack->start + n;
    stack->p = stack->end;
}
static void stackFree(struct Stack *stack) {
    free(stack->start);
}
static inline void stackPush(struct Stack *stack, uint elem) {
    *(--stack->p) = elem;
}
static inline uint stackPop(struct Stack *stack) {
    return *(stack->p++);
}
static inline void stackClear(struct Stack *stack) {
    stack->p = stack->end;
}
static inline bool stackIsEmpty(struct Stack *stack) {
    return stack->p == stack->end;
}
static inline uint stackPeek(struct Stack *stack) {
    return *(stack->p);
}