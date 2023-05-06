import sys
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def def_args():
    parser = argparse.ArgumentParser(description='图像去雾')
    parser.add_argument('-i', '--input', default='images/2.png')
    parser.add_argument('-o', '--output', default='images/result.png')

    parser.add_argument('--save_I_dark', default=True, help='是否保存暗通道图')
    parser.add_argument('--save_t', default=True, help='是否保存透射率图')
    parser.add_argument('--save_J_dark', default=True, help='是否保存生成图的暗通道图')

    parser.add_argument('--window_size', default=15, help='窗口大小需为奇数。窗口越大，越满足暗通道先验，但边缘越明显')
    parser.add_argument('--omiga', default=0.95, help='保留雾的权重，越小保留的越多')
    parser.add_argument('--t0', default=0.1, help='防止图像过白或天空部分对比度奇怪。当图像整体偏亮，值应设置越大')
    parser.add_argument('--eps', default=50, help='对透射率的引导滤波参数，值越大，平滑程度越高')

    return parser.parse_args()


def calcu_dark_channel(I, window_size=3):
    assert window_size % 2 == 1
    ws = window_size
    pad = ws // 2
    I_channel_dark = np.minimum(I[:, :, 0], I[:, :, 1])
    I_channel_dark = np.minimum(I_channel_dark, I[:, :, 2])
    h, w = I_channel_dark.shape
    I_large = np.pad(I_channel_dark, ((pad, pad), (pad, pad)), mode='edge')
    for i in range(ws):
        for j in range(ws):
            I_channel_dark = np.minimum(I_channel_dark, I_large[i:i + h, j:j + w])
    return I_channel_dark


def calcu_A(I, I_dark):
    I_dark_flat = I_dark.flatten()
    sorted_idx = np.argsort(I_dark_flat)
    num_pixels = int(len(sorted_idx) * 0.001)
    top_sorted_idx = sorted_idx[-num_pixels:]
    I_values = I.reshape([-1, 3])[top_sorted_idx]
    A = np.max(I_values, 0)
    return A


def calcu_t(I, A):
    '''
    计算图像透射率
    :param I: 输入图像 [h,w,3]
    :param A: 大气成分 [3]
    :return: 透射率 t:[h,w]
    '''
    I_A = I / A.reshape(1, 1, -1)
    I_A_dark = calcu_dark_channel(I_A, window_size=args.window_size)
    t = 1. - args.omiga * I_A_dark
    guide = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)[:, :, 0]
    t = cv2.ximgproc.guidedFilter(guide=guide, src=t, radius=8, eps=args.eps, dDepth=-1)
    # t = gaussian_filter(t, args.sigma)
    t = np.maximum(t, args.t0)
    return t


def calcu_J(I, A, t):
    '''
    图像去雾
    :param I: [h,w,3]
    :param A: [3]
    :param t: [h,w]
    :return: J[h,w,3]
    '''
    A = A.reshape([1, 1, -1])
    t = t.reshape([t.shape[0], t.shape[1], 1])
    J = (I - A) / t + A
    return J


def main():
    # 输入图像 I:[h,w,3]
    img = cv2.imread(args.input)
    I = np.array(img).astype('float32')

    # 计算暗通道图 I_dark:[h,w]
    I_dark = calcu_dark_channel(I, window_size=args.window_size)
    if args.save_I_dark:
        I_dark = np.round(I_dark).astype('uint8')
        cv2.imwrite('images/dark.png', I_dark, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # 根据原图和暗通道图计算大气成分 A:[3]
    A = calcu_A(I, I_dark)
    print(f'A为{A}')
    # A = np.array([255, 255, 255])

    # 根据原图和A计算透射率 t:[h,w]
    t = calcu_t(I, A)
    if args.save_I_dark:
        plt.imshow(t, cmap='gray')
        plt.savefig('images/t.png', dpi=1000)

    # 根据原图、A和t计算处理后图像 J:[h,w,3]
    J = calcu_J(I, A, t)
    print(f'计算的J的范围为{np.min(J)}到{np.max(J)}')
    J[J < 0] = 0.
    J[J > 255] = 255.

    # 根据生成图J，计算其暗通道图
    J_dark = calcu_dark_channel(J, window_size=args.window_size)
    if args.save_J_dark:
        J_dark = np.round(J_dark).astype('uint8')
        cv2.imwrite('images/J_dark.png', J_dark, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    J = np.round(J).astype('uint8')
    cv2.imwrite(args.output, J, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    args = def_args()
    main()
