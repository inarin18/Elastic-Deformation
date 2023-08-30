# ######################################################################################################
#
#  手書き数字の画像を人為的に歪ませる．
#
# 「方針」
#   変位場を作成してそれをもとに元の画像を歪ませる．
#   
#   1. まず画像の各ピクセルごとにランダムな変位 \Delta x, \Delta y \in (0, 1)
#      を生成する．
#
#   2. その後，その変位場を様々な分散のガウシアンフィルタと畳み込み，平滑化する．
#
#   3. 平滑化した変位場を用いて手書き数字の画像を歪ませる．
#      (この際，テキストには記載されていないが変位場をスケール係数 \alpha を用いて拡大しておく必要がある)
#
# #####################################################################################################

from PIL import Image
import numpy as np


def generate_initial_displacement_field(num_of_rows: int, num_of_clms: int) -> np.ndarray:
    
    # rows x clms x 2 のサイズの3次元配列を0で初期化
    dsplcmnt_fld = np.zeros((num_of_rows, num_of_clms, 2))
    
    # 変位 \Delta x, \Delta y を (0, 1) の乱数で生成
    for i in range(num_of_rows):
        for j in range(num_of_clms): 
            dsplcmnt_fld[i][j][0] = 2 * np.random.rand() - 1  # \Delta x
            dsplcmnt_fld[i][j][1] = 2 * np.random.rand() - 1  # \Delta y
    
    return dsplcmnt_fld


# ここで x, y は独立であるとする．このとき共分散行列は単位行列に比例する．
def gaussian_distribution(x: float, y: float, sigma: float, mean_x: float = 0, mean_y: float = 0) -> float:
    
    normalization_coeffcient = 1 / (2*np.pi*sigma**2)

    return normalization_coeffcient * np.exp(- ((x - mean_x)**2 + (y - mean_y)**2) / (2*sigma**2))


def generate_gaussian_filter(sigma: float) -> np.ndarray:
    
    # サイズの決定 ( 幅 = 分散 と解釈 ) ; 分散が１より小さい場合は 3 x 3 のフィルタとする．
    size = int(4 * sigma + 1) if sigma >= 1 else 3
    
    # サイズをもとに平均(中心)を決定
    mean_x = mean_y = (size - 1) / 2
    
    # ガウス分布をもとにフィルタを作成
    filter = np.array([[gaussian_distribution(x=x, 
                                              y=y, 
                                              sigma=sigma, 
                                              mean_x=mean_x, 
                                              mean_y=mean_y) 
                        for x in range(size)] 
                       for y in range(size)] 
                      )
    
    normalized_filter = filter / np.sum(filter)
    
    return normalized_filter


# 変位場の平滑化を標準偏差 \sigma のガウシアンフィルタを畳み込むことで行う
def smooth_displacement_filed_with_gaussian(dsplcmnt_fld: np.ndarray, sigma: float) -> np.ndarray:
    
    # 行数，列数を取得
    num_of_rows, num_of_clms = len(dsplcmnt_fld), len(dsplcmnt_fld[0])
    
    # 変位場を平滑化して得られる変位を格納する配列を用意
    smoothed_fld = [[[0, 0] for _ in range(num_of_clms)] for _ in range(num_of_rows)] 
    
    # フィルタを生成
    gaussian_filter = generate_gaussian_filter(sigma)
    size_of_filter  = len(gaussian_filter)
    center_of_filer = int((size_of_filter - 1) / 2)
    
    # 変位場の各座標に対しフィルタを畳み込み
    for i in range(num_of_rows):
        for j in range(num_of_clms):
            # 畳み込み和の初期化
            convolutional_sum_x = 0
            convolutional_sum_y = 0
            # 畳み込み演算
            for u in range(size_of_filter):
                for v in range(size_of_filter):
                    try :
                        convolutional_sum_x += dsplcmnt_fld[i-center_of_filer+u][j-center_of_filer+v][0] * gaussian_filter[u][v]
                        convolutional_sum_y += dsplcmnt_fld[i-center_of_filer+u][j-center_of_filer+v][1] * gaussian_filter[u][v]
                    except IndexError:
                        convolutional_sum_x += 0
                        convolutional_sum_y += 0
                        
            smoothed_fld[i][j][0] = convolutional_sum_x
            smoothed_fld[i][j][1] = convolutional_sum_y
        
    smoothed_fld = np.array(smoothed_fld)
            
    return smoothed_fld


# 境界処理を施してピクセル値を返す
def get_value_of(array, i, j):
    
    i_max, j_max = len(array), len(array[0])
    
    if 0 <= i < i_max and 0 <= j < j_max:
        return array[i][j]
    else :
        return 0
    


# 新たなピクセル値をバイリニア補間により導出
def compute_pixel_with_bilinear_interpolation(x, y, delta_x, delta_y, img_array: np.ndarray) -> int:
    
    """ Reference : https://en.wikipedia.org/wiki/Bilinear_interpolation """
    
    x_displaced, y_displaced = x + delta_x, y + delta_y 
    
    x1, x2 = int(np.floor(x_displaced)), int(np.ceil(x_displaced))
    y1, y2 = int(np.floor(y_displaced)), int(np.ceil(y_displaced))
    
    img_xd_y1 = ((x2 - x_displaced) / (x2 - x1)) * get_value_of(array=img_array, i=x1, j=y1) \
              + ((x_displaced - x1) / (x2 - x1)) * get_value_of(array=img_array, i=x2, j=y1)
              
    img_xd_y2 = ((x2 - x_displaced) / (x2 - x1)) * get_value_of(array=img_array, i=x1, j=y2) \
              + ((x_displaced - x1) / (x2 - x1)) * get_value_of(array=img_array, i=x2, j=y2)
                
    img_xd_yd = ((y2 - y_displaced) / (y2 - y1)) * img_xd_y1 + ((y_displaced - y1) / (y2 - y1)) * img_xd_y2
    
    return img_xd_yd



def warp_image_with_displacement_field(img_array: np.ndarray, dsplcmnt_fld: np.ndarray) -> np.ndarray:
    
    size_of_img = len(img_array)
    
    new_img_array = [[ 0  for _ in range(size_of_img)] for _ in range(size_of_img)]
    
    for i in range(size_of_img):
        for j in range(size_of_img):
            
            delta_x, delta_y = dsplcmnt_fld[i][j][0], dsplcmnt_fld[i][j][1]
            
            new_img_array[i][j] = compute_pixel_with_bilinear_interpolation(i, 
                                                                            j, 
                                                                            delta_x, 
                                                                            delta_y, 
                                                                            img_array
                                                                            )
    return np.array(new_img_array)       
            
   

if __name__ == "__main__":
    
    debug = 0
    
    np.random.seed(seed=64)
    
    handwritten_digits_raw_data_path = "./Data/qmnist-train-images-idx3-ubyte"
    
    with open(handwritten_digits_raw_data_path, "rb") as f:
        # バイナリのピクセル値以外の情報を取得
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_of_imgs  = int.from_bytes(f.read(4), byteorder="big")
        num_of_rows  = int.from_bytes(f.read(4), byteorder="big") 
        num_of_clms  = int.from_bytes(f.read(4), byteorder="big")
        
        # ｎ枚目の画像のピクセル値を取得し2d-arrayに格納
        for _ in range(4):
            img_array_ = np.array([[int.from_bytes(f.read(1), byteorder="big") for _ in range(num_of_clms)] for _ in range(num_of_rows)])
        img_array = np.array([[int.from_bytes(f.read(1), byteorder="big") for _ in range(num_of_clms)] for _ in range(num_of_rows)])
 
        img = Image.fromarray(img_array, mode="F")
        img.show()
        
        dsplcmnt_fld = generate_initial_displacement_field(num_of_rows, num_of_clms)
        
        alpha = 100           # scale factor
        sigma = np.sqrt(20)   # standard deviation
        
        smoothed_fld = smooth_displacement_filed_with_gaussian(dsplcmnt_fld, sigma=sigma)
        
        scaled_smoothed_fld = alpha * smoothed_fld
        
        warped_img_array = warp_image_with_displacement_field(img_array=img_array, dsplcmnt_fld=scaled_smoothed_fld)
        
        warped_img = Image.fromarray(warped_img_array)
        warped_img.show()
    
        
        if debug:
            
            print(dsplcmnt_fld)
            
            gaussian_filter = generate_gaussian_filter(sigma=sigma)
            print(gaussian_filter)
            
            print(smoothed_fld)
        