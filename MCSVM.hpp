#ifndef MCSVM_H_INCLUDED
#define MCSVM_H_INCLUDED

#include "SVM.hpp"

// MCSVMの学習状態
typedef enum {
  MCSVM_NOT_LEARN,        // 学習していない
  MCSVM_LEARN_SUCCESS,    // 正常に全SVMを学習した.
} MCSVM_STATUS;

// クラスi,jを識別するSVMのインデックスを返す
#define INX_KSVM_IJ(n_class,i,j) ((((i) * ((2 * (n_class)) - (3) - (i)))/2) + ((j) - (1)))

// Class Multi-Class SVM
// One-vs-One法(うーんこの)によるSVM.

class MCSVM : public SVM
{
  private:
    int    n_class;      // 識別クラス数
                         // 異なる2クラスi,j(i < j)を識別するSVMをインデックスi*(2k-3-i)/2 + j-1で参照する
    int    maxFailcount; // 学習失敗を許容する最大回数
    
    // マルチクラス用の拡張:サンプルは共有し, 各SVMのパラメタを個別に保持
    float* mc_alpha;     // 各識別用の双対係数
    int*   mc_label;     // 各識別用の2値(-1,1)ラベル, 識別に関係しないデータにはラベル0が付与される.
                         // マルチクラス識別の場合,SVM::labelには0,...,n_class-1までのラベルが付いている
  public:
    MCSVM(int,      // クラス個数
          int,      // データ次元
          int,      // サンプル個数
          float*,   // サンプルデータ
          int*);    // マルチクラスラベル
          
    ~MCSVM(void);
          
    // 未知データのラベルを推定する.返り値はマルチクラスラベル0,...,n_class-1
    virtual int predict_label(float*);
    
    // 未知データの識別確率を推定する.
    // ラベル識別predict_label結果の整合性を考えない.
    virtual float predict_probability(float*);

    // 全てのSVMの学習する.
    virtual int learning(void);
    
    // 双対係数のゲッター
    virtual float* get_alpha(void);
    
    // 双対係数/ラベルのセッター
    void set_alpha(float*, int, int);
                        
};

#endif /* MCSVM_H_INCLUDED */
