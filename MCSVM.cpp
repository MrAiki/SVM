#include "MCSVM.hpp"

// コンストラクタ. 適宜追加予定
MCSVM::MCSVM(int    class_n,
             int    sample_dim,
             int    sample_n, 
             float* sample_data,
             int*   sample_mclabel)
  : SVM(sample_dim, sample_n, sample_data, sample_mclabel)
{
  this->n_class = class_n;
  this->maxFailcount = 5;

  int n_kC2 = (n_class) * (n_class - 1) / 2;
  
  mc_alpha   = new float[n_sample * n_kC2];
  mc_label   = new int[n_sample * n_kC2];

  // ラベルの生成
  int tmp_lab;
  for (int ci = 0; ci < n_class; ci++) {
    for (int cj = ci + 1; cj < n_class; cj++) {
      // i < jであることから, ラベルiは負例, ラベルjは正例に割り当てる.
      // いずれのラベルにも該当しないデータは欠損とし,ラベル0とする.
      for (int l=0; l < n_sample; l++) {
        if (label[l] == ci) {
          tmp_lab = -1;
        } else if (label[l] == cj) {
          tmp_lab = 1;
        } else {
          tmp_lab = 0;
        }
        MATRIX_AT(mc_label,n_sample,INX_KSVM_IJ(n_class,ci,cj),l) = tmp_lab;
        printf("l %d : %d -> %d \r\n", l, label[l], tmp_lab);
      }

    }
  }

}

// 領域開放
MCSVM::~MCSVM(void)
{
    delete [] mc_alpha;
    delete [] mc_label;
}

// 全SVMの学習.
int MCSVM::learning(void)
{
  int status, fail_count;
  int* tmp_label = new int[n_sample];
  // 元のラベルを退避
  memcpy(tmp_label,label, sizeof(int) * n_sample);
  for (int ci = 0; ci < n_class; ci++) {
    for (int cj = ci + 1; cj < n_class; cj++) {
      // 2値ラベルを取得する.
      memcpy(label, &(MATRIX_AT(mc_label, n_sample, INX_KSVM_IJ(n_class,ci,cj), 0)), sizeof(int) * n_sample);
      // 学習 - 学習失敗の場合はリトライする.
      fail_count = 0;
      do {
        if (fail_count >= maxFailcount) {
          fprintf(stderr, "Learning failed %d times at %d,%d classifier, give up \r\n", fail_count, ci, cj);
          return MCSVM_NOT_LEARN;
        }
        status = SVM::learning();

        if ( (status == SVM_NOT_CONVERGENCED)
            || (status == SVM_DETECT_BAD_VAL) ) {
          fail_count++;
        }
      } while (status != SVM_LEARN_SUCCESS);
      // 学習結果の係数とSVラベルの取得
      memcpy(&(MATRIX_AT(mc_alpha, n_sample, INX_KSVM_IJ(n_class,ci,cj), 0)), this->alpha, sizeof(float) * n_sample);
    }
  }

  // 元のラベルを復帰
  memcpy(this->label, tmp_label, sizeof(int) * n_sample);
  delete [] tmp_label;
  return MCSVM_LEARN_SUCCESS;

}

// 未知データのラベルを推定する.
int MCSVM::predict_label(float* data) 
{
  // 単位ステップ関数による決定的識別
  float net;
  int* result_label_count = new int[n_class];

  int tmp_ci, tmp_cj;
  memset(result_label_count, 0, sizeof(int) * n_class);
  for (int ci = 0; ci < n_class; ci++) {
    for (int cj = ci + 1; cj < n_class; cj++) {

      // インデックスをi < jに
      tmp_ci = ci; tmp_cj = cj;
      if ( ci > cj ) {
        tmp_cj = ci; tmp_ci = cj;
      }
      // 係数とラベルを取得し,ci,cjを識別するSVMを構成
      memcpy(this->alpha, &(MATRIX_AT(mc_alpha, n_sample, INX_KSVM_IJ(n_class,tmp_ci,tmp_cj), 0)), sizeof(float) * n_sample);
      memcpy(this->label, &(MATRIX_AT(mc_label, n_sample, INX_KSVM_IJ(n_class,tmp_ci,tmp_cj), 0)), sizeof(float) * n_sample);

      // 識別:識別されたクラスに投票.
      net = SVM::predict_net(data);
      //printf("ci:%d cj:%d >> net : %f \n", ci, cj, net);
      if ( net < 0 ) {
        result_label_count[ci]++;
      } else if ( net >= 0 ) {
        result_label_count[cj]++;
      }

    }
    //printf("sum_net[%d] : %f \n", ci, sum_net[ci]);
  }

  // 判定:最大頻度のクラスに判定する.
  int max,argmax;
  max = 0;
  for (int i = 0; i < n_class; i++) {
    //printf("%d : %d \n", i, result_label_count[i]);
    if ( result_label_count[i] > max ) {
      max = result_label_count[i];
      argmax = i;
    }
  }

  delete [] result_label_count;
  return argmax;

}

// 未知データの識別確率を推定する.
float MCSVM::predict_probability(float* data) 
{
  // シグモイド関数による確率的識別
  float prob;
  float* result_label_prob = new float[n_class];
  int tmp_ci, tmp_cj;
  memset(result_label_prob, 0, sizeof(float) * n_class);
  for (int ci = 0; ci < n_class; ci++) {
    for (int cj = ci + 1; cj < n_class; cj++) {

      // インデックスをci < cjに : 負例はci, 正例はcj
      tmp_ci = ci; tmp_cj = cj;
      if ( ci > cj ) {
        tmp_cj = ci; tmp_ci = cj;
      }
      // 係数とラベルを取得し,ci,cjを識別するSVMを構成
      memcpy(this->alpha, &(MATRIX_AT(mc_alpha, n_sample, INX_KSVM_IJ(n_class,tmp_ci,tmp_cj), 0)), sizeof(float) * n_sample);
      memcpy(this->label, &(MATRIX_AT(mc_label, n_sample, INX_KSVM_IJ(n_class,tmp_ci,tmp_cj), 0)), sizeof(float) * n_sample);

      // 確率識別:確率の足し上げ
      prob = SVM::predict_probability(data);
      if ( prob > float(0.5) ) {
          result_label_prob[cj] += prob;
      } else {
          result_label_prob[ci] += (1-prob);
      }

    }
    //printf("sum_net[%d] : %f \n", ci, sum_net[ci]);
  }

  // 判定:最大確率和
  // おそらくラベル識別との整合性は取れる...はず
  float max = 0;
  for (int i = 0; i < n_class; i++) {
    //printf("%d : %d \n", i, result_label_count[i]);
    if ( result_label_prob[i] > max ) {
      max = result_label_prob[i];
    }
  }

  delete [] result_label_prob;
  // 平均確率を返す.
  return (max / (n_class-1));

}

// override
float* MCSVM::get_alpha(void) {
    return (float *)mc_alpha;
}

// override
void MCSVM::set_alpha(float* mcalpha_data, int nsample, int nclass) {
    if ( nsample != n_sample ) {
        fprintf( stderr, " set_alpha : number of sample isn't match : n_samle= %d, arg= %d \r\n", n_sample, nsample);
        return;
    } else if ( nclass != n_class ) {
        fprintf( stderr, " set_alpha : number of class isn't match : n_class= %d, nclass= %d \r\n", n_class, nclass);
        return;
    }
    int nC2 = n_class * (n_class - 1)/2;
    memcpy(mc_alpha, mcalpha_data, sizeof(float) * n_sample * nC2);
    status = SVM_SET_ALPHA;
}
