#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#include "mbed.h"

/* float.h がないので, FLT_MAXをここで */
#define FLT_MAX  (0x1.fffffeP127F)      // float max

// 特に行列演算用のマクロ集

// 行列アクセス用マクロ. n_lenは行列の幅(length, col,列の数)
#define MATRIX_AT(ary,n_len,i,j) (ary[(((i) * (n_len)) + (j))])

// 2乗(ユーグリッド)ノルムを返す
inline float two_norm(float *vec, int dim) {
  register float ret = 0;
  for (int i=0; i < dim; i++) {
    ret += powf(vec[i],2);
  }
  return sqrtf(ret);
}

// 2ベクトル間の距離をユーグリッドノルムで測る.
inline float vec_dist(float *x, float *y, int dim) {
  register float ret = 0;
  for (int i=0; i < dim; i++) {
    ret += powf(x[i] - y[i],2);
  }
  return sqrtf(ret);
}

// 一様乱数の生成 : [-w,w]で生成.
inline float uniform_rand(float w) {
  return (float(rand() - RAND_MAX/2) / float(RAND_MAX)) * 2 * w;
}

// float配列の最大値を返す
inline float maxf(float* ary, int dim) {
  register float max = 0;
  for (int i=0; i < dim; i++) {
    if (ary[i] >= max) {
      max = ary[i];
    }
  }
  return max;
}

// float配列の最小値を返す
inline float minf(float* ary, int dim) {
  register float min = FLT_MAX;
  for (int i=0; i < dim; i++) {
    if (ary[i] <= min) {
      min = ary[i];
    }
  }
  return min;
}

// サイズm*nの行列とサイズn*1のベクトルの掛け算を計算し,結果をresultにセットする.

inline void multiply_mat_vec(float*  mat, // m * n 行列
                             float*  vec, // n * 1 ベクトル
                             float*  result, // m * 1 計算結果ベクトル
                             int     m,   // m
                             int     n)   // n
{
  register float sum;
  for (int i=0;i<m;i++) {
    sum = 0;
    for (int j=0;j<n;j++) {
      sum += MATRIX_AT(mat,n,i,j) * vec[j];
    }
    result[i] = sum;
  }
}

// シグモイド（ロジスティック）関数.
inline float sigmoid_func(float x){
  return (1 / (1 + expf(-x)));
}

// 信号の正規化([0,1]の範囲に収めること:超重要)
inline float normalize_signal(float in, float max, float min){
  return float((in - min) / (max - min));
} 

// 信号の元の領域への拡大([0,1]から元の信号値にスケール)
inline float expand_signal(float in, float max, float min){
  return float(min + (max - min) * in);
} 

#endif /* UTIL_H_INCLUDED */
