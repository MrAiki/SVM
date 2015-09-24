#include "SVM.hpp"

SVM::SVM(int dim_sample, int n_sample, float* sample_data, int* sample_label)
{
  this->dim_signal = dim_sample;
  this->n_sample = n_sample;

  // 各配列(ベクトル)のアロケート
  alpha   = new float[n_sample];
  // grammat = new float[n_sample * n_sample];
  label   = new int[n_sample];
  sample  = new float[dim_signal * n_sample];
  sample_max = new float[dim_signal];
  sample_min = new float[dim_signal];

  // サンプルのコピー
  memcpy(this->sample, sample_data,
          sizeof(float) * dim_signal * n_sample);
  memcpy(this->label, sample_label,
          sizeof(int) * n_sample);

  // 正規化のための最大最小値
  memset(sample_max, 0, sizeof(float) * dim_sample);
  memset(sample_min, 0, sizeof(float) * dim_sample);
  for (int i = 0; i < dim_signal; i++) {
    float value;
    sample_min[i] = FLT_MAX;
    for (int j = 0; j < n_sample; j++) {
      value = MATRIX_AT(this->sample, dim_signal, j, i);
      if ( value > sample_max[i] ) {
        sample_max[i] = value;
      } else if ( value < sample_min[i] ) {
        //printf("min[%d] : %f -> ", i, sample_min[i]);
        sample_min[i] = value;
        //printf("min[%d] : %f \dim_signal", i, value);
      }
    }
  }

  // 信号の正規化 : 死ぬほど大事
  for (int i = 0; i < dim_signal; i++) {
    float max,min;
    max = sample_max[i];
    min = sample_min[i];
    for (int j = 0; j < n_sample; j++) {
      //printf("[%d,%d] %f -> ", i, j, MATRIX_AT(this->sample, dim_signal, j, i));
      MATRIX_AT(this->sample, dim_signal, j, i) = ( MATRIX_AT(this->sample, dim_signal, j, i) - min ) / (max - min);
      //printf("%f\dim_signal", MATRIX_AT(this->sample, dim_signal, j, i));
    }
  }

  /* // グラム行列の計算 : メモリの制約上,廃止
  for (int i = 0; i < n_sample; i++) {
    for (int j = i; j < n_sample; j++) {
      MATRIX_AT(grammat,n_sample,i,j) = kernel_function(&(MATRIX_AT(this->sample,dim_signal,i,0)), &(MATRIX_AT(this->sample,dim_signal,j,0)), dim_signal);
      // グラム行列は対称
      if ( i != j ) {
        MATRIX_AT(grammat,n_sample,j,i) = MATRIX_AT(grammat,n_sample,i,j);
      }
    }
  }
  */

  // 学習関連の設定. 例によって経験則
  this->maxIteration = 5000;
  this->epsilon      = float(0.00001);
  this->eta          = float(0.05);
  this->learn_alpha  = float(0.8) * this->eta;
  this->status       = SVM_NOT_LEARN;

  // ソフトマージンの係数. 両方ともFLT_MAXとすることでハードマージンと(ほぼ)一致.
  // また, 設定するときはどちらか一方のみにすること.
  C1 = FLT_MAX;
  C2 = 5;

  srand((unsigned int)time(NULL));

}

// 楽園追放
SVM::~SVM(void) 
{
    delete [] alpha; delete [] label;
    delete [] sample;
    delete [] sample_max; delete [] sample_min;
}

// 再急勾配法（ｻｰｾﾝwww）による学習
int SVM::learning(void)
{

  int iteration;              // 学習繰り返しカウント

  float* diff_alpha;          // 双対問題の勾配値
  float* pre_diff_alpha;      // 双対問題の前回の勾配値（慣性項に用いる）
  float* pre_alpha;           // 前回の双対係数ベクトル（収束判定に用いる）
  register float diff_sum;    // 勾配計算用の小計
  register float kernel_val;  // カーネル関数とC2を含めた項

  //float plus_sum, minus_sum;  // 正例と負例の係数和

  // 配列（ベクトル）のアロケート
  diff_alpha     = new float[n_sample];
  pre_diff_alpha = new float[n_sample];
  pre_alpha      = new float[n_sample];

  status = SVM_NOT_LEARN;       // 学習を未完了に
  iteration  = 0;       // 繰り返し回数を初期化

  // 双対係数の初期化.乱択
  for (int i = 0; i < n_sample; i++ ) {
    // 欠損データの係数は0にして使用しない
    if ( label[i] == 0 ) {
      alpha[i] = 0;
      continue;
    }
    alpha[i] = uniform_rand(1.0) + 1.0;
  }

  // 学習ループ
  while ( iteration < maxIteration ) {

    printf("ite: %d diff_norm : %f alpha_dist : %f \r\n", iteration, two_norm(diff_alpha, n_sample), vec_dist(alpha, pre_alpha, n_sample));
    // 前回の更新値の記録
    memcpy(pre_alpha, alpha, sizeof(float) * n_sample);
    if ( iteration >= 1 ) {
      memcpy(pre_diff_alpha, diff_alpha, sizeof(float) * n_sample);
    } else {
      // 初回は0埋めで初期化
      memset(diff_alpha, 0, sizeof(float) * n_sample);
      memset(pre_diff_alpha, 0, sizeof(float) * n_sample);
    }

    // 勾配値の計算
    for (int i=0; i < n_sample; i++) {
      diff_sum = 0;
      for (int j=0; j < n_sample;j++) {
        // C2を踏まえたカーネル関数値
        kernel_val = kernel_function(&(MATRIX_AT(sample,dim_signal,i,0)), &(MATRIX_AT(sample,dim_signal,j,0)), dim_signal);
        // kernel_val = MATRIX_AT(grammat,n_sample,i,j); // via Gram matrix
        if (i == j) { 
          kernel_val += (1/C2);
        }
        diff_sum += alpha[j] * label[j] * kernel_val; 
      }
      diff_sum *= label[i];
      diff_alpha[i] = 1 - diff_sum;
    }

    // 双対変数の更新
    for (int i=0; i < n_sample; i++) {
      if ( label[i] == 0 ) {
        continue;
      }
      //printf("alpha[%d] : %f -> ", i, alpha[i]);
      alpha[i] = pre_alpha[i] 
                  + eta * diff_alpha[i]
                  + learn_alpha * pre_diff_alpha[i];
      //printf("%f \dim_signal", alpha[i]);

      // 非数/無限チェック
      if ( isnan(alpha[i]) || isinf(alpha[i]) ) {
        fprintf(stderr, "Detected NaN or Inf Dual-Coffience : pre_alhpa[%d]=%f -> alpha[%d]=%f", i, pre_alpha[i], i, alpha[i]);
        return SVM_DETECT_BAD_VAL;
      }

    }

    // 係数の制約条件1:正例と負例の双対係数和を等しくする.
    //                 手法:標本平均に寄せる
    float norm_sum = 0;
    for (int i = 0; i < n_sample; i++ ) {
      norm_sum += (label[i] * alpha[i]);
    }
    norm_sum /= n_sample;

    for (int i = 0; i < n_sample; i++ ) {
      if ( label[i] == 0 ) {
        continue;
      }
      alpha[i] -= (norm_sum / label[i]);
    }

    // 係数の制約条件2:双対係数は非負
    for (int i = 0; i < n_sample; i++ ) {
      if ( alpha[i] < 0 ) {
        alpha[i] = 0;
      } else if ( alpha[i] > C1 ) {
        // C1を踏まえると,係数の上限はC1となる.
        alpha[i] = C1;
      }  
    }

    // 収束判定 : 凸計画問題なので,収束時は大域最適が
    //            保証されている.
    if ( (vec_dist(alpha, pre_alpha, n_sample) < epsilon)
        || (two_norm(diff_alpha, n_sample) < epsilon) ) {
      // 学習の正常完了
      status = SVM_LEARN_SUCCESS;
      break;
    }

    // 学習繰り返し回数のインクリメント
    iteration++;
  }

  if (iteration >= maxIteration) {
    fprintf(stderr, "Learning is not convergenced. (iteration count > maxIteration) \r\n");
    status = SVM_NOT_CONVERGENCED;
  } else if ( status != SVM_LEARN_SUCCESS ) {
    status = SVM_NOT_LEARN;
  }
  
  // 領域開放
  delete [] diff_alpha;
  delete [] pre_diff_alpha;
  delete [] pre_alpha;
  
  return status;

}

// 未知データのネットワーク値を計算
float SVM::predict_net(float* data)
{
  // 学習の終了を確認
  if (status != SVM_LEARN_SUCCESS && status != SVM_SET_ALPHA) {
    fprintf(stderr, "Learning is not completed yet.");
    //exit(1);
    return SVM_NOT_LEARN;
  }

  float* norm_data = new float[dim_signal];

  // 信号の正規化
  for (int i = 0; i < dim_signal; i++) {
    norm_data[i] = ( data[i] - sample_min[i] ) / ( sample_max[i] - sample_min[i] );
  }

  // ネットワーク値の計算
  float net = 0;
  for (int l=0; l < n_sample; l++) {
    // **係数が正に相当するサンプルはサポートベクトル**
    if(alpha[l] > 0) {
      net += label[l] * alpha[l]
              * kernel_function(&(MATRIX_AT(sample,dim_signal,l,0)), norm_data, dim_signal);
    }
  }

  return net;

}

// 未知データの識別確率を計算
float SVM::predict_probability(float* data)
{
    float net, probability;
    float* optimal_w = new float[dim_signal];   // 最適時の係数(not 双対係数)
    float sigmoid_param;                        // シグモイド関数の温度パラメタ
    float norm_w;                               // 係数の2乗ノルム
    
    net = SVM::predict_net(data);
    
    // 最適時の係数を計算
    for (int n = 0; n < dim_signal; n++ ) {
        optimal_w[n] = 0;
        for (int l = 0; l < n_sample; l++ ) {
            optimal_w[n] += alpha[l] * label[l] * MATRIX_AT(sample, dim_signal, l, n);
        }
    }
    norm_w = two_norm(optimal_w, dim_signal);
    sigmoid_param = 1 / ( norm_w * logf( (1 - epsilon) / epsilon ) );
    
    probability = sigmoid_func(net/sigmoid_param);
    
    delete [] optimal_w;
    
    // 打ち切り:誤差epsilon以内ならば, 1 or 0に打ち切る.
    if ( probability > (1 - epsilon) ) {
        return float(1);
    } else if ( probability < epsilon ) {
        return float(0);
    }
    
    return probability;
    
}

// 未知データの識別
int SVM::predict_label(float* data)
{
  return (predict_net(data) >= 0) ? 1 : (-1);
}

// 双対係数のゲッター
float* SVM::get_alpha(void) {
    return (float *)alpha;
}

// 双対係数のセッター
void SVM::set_alpha(float* alpha_data, int nsample) {
    if ( nsample != n_sample ) {
        fprintf( stderr, " set_alpha : number of sample isn't match with arg. n_samle= %d, arg= %d \r\n", n_sample, nsample);
        return;
    }
    memcpy(alpha, alpha_data, sizeof(float) * nsample);
    status = SVM_SET_ALPHA;
}