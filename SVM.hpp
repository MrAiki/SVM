#ifndef SVM_H_INCLUDED
#define SVM_H_INCLUDED

#include "mbed.h"

#include "util.hpp"

// SVMの学習状態
typedef enum {
  SVM_NOT_LEARN,        // 学習していない
  SVM_LEARN_SUCCESS,    // 正常終了(収束によって終了した)
  SVM_NOT_CONVERGENCED, // 繰り返し上限到達
  SVM_DETECT_BAD_VAL,   // 係数で非数/無限を検知した
  SVM_SET_ALPHA,        // 学習していないが,最適化した係数がセットされている
} SVM_STATUS;

// Class SVM

class SVM
{
  protected:

    int dim_signal;     // 入力データの次元
    int n_sample;       // サンプルの個数
    float* sample_max;  // 特徴の正規化用の,各次元の最大値
    float* sample_min;  // 特徴の正規化用の,各次元の最小値
    float* alpha;       // 双対係数のベクトル
    // float* grammat;     // グラム（カーネル）行列 -> 領域をn_sample^2食うので,廃止.時間はかかるけど逐次処理. 
    int maxIteration;   // 学習の最大繰り返し回数
    float epsilon;      // 収束判定用の小さな値
    float eta;          // 学習係数
    float learn_alpha;  // 慣性項の係数
    float C1;           // 1ノルムソフトマージンンのスラック変数とハードマージンのトレード
                        // オフを与える係数 (無限でハードマージンに一致,FLT_MAXで近似)
    float C2;           // 2ノルムソフトマージンの〜,（無限でハードマージンに一致,FLT_MAXで近似）
                        // また,C1かC2どちらか一つのみを設定すること.

  public:
    int*   label;       // サンプルの2値ラベルの配列(-1 or 1)
    float* sample;      // n次元サンプルデータの配列.
    int    status;      // SVMの状態

  protected:
    // カーネル関数. ここでは簡易なRBFカーネルをハードコーディング
    inline float kernel_function(float *x, float *y, int n) {
      register float inprod = 0;
      for (int i=0;i < n;i++) {
        inprod += powf(x[i] - y[i],2);
      }
      return expf(-inprod/0.1);
    }

  public:
    // 最小の引数による初期化. 順次拡張予定.
    SVM(int,      // データ次元
        int,      // サンプル個数
        float*,   // サンプル
        int*);    // ラベル
        
    ~SVM(void);

    // 学習によりマージンを最大化し,サポートベクトルを確定させる.
    virtual int learning(void);

    // 未知データの識別.データを受け取り,識別ラベルを-1 or 1で返す.
    virtual int predict_label(float*);

    // 未知データのネットワーク値（負の値ならば0,即ち識別面の下半空間に,
    // 正の値ならば1,識別面の上半空間に存在すると判定）を計算して返す.
    float predict_net(float*);
    
    // 未知データの正例の識別確率[0,1]を返す.
    // 予測はシグモイド関数による.
    // 1ならばマージンを超えて完全に正例領域に入っている.
    // 0ならばマージンを超えて完全に負例領域に入っている.
    virtual float predict_probability(float*);
    
    // 双対係数のゲッター
    virtual float* get_alpha(void);
    
    // 双対係数のセッター
    virtual void set_alpha(float*, int);

};

#endif /* SVM_H_INCLUDED */
