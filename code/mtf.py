import math as math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pylab as pylab
import scipy as scipy
from scipy import ndimage
from scipy import interpolate
from scipy.fft import fft
from scipy.optimize import curve_fit
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import scipy.stats
import datetime
warnings.filterwarnings('ignore')

########################################################################################################################


def f1(x, a1, b1, c1):
    return a1 / (1.0 + np.exp((x-b1)/c1))


def f2(x, a2, b2, c2):
    return a2 / (1.0 + np.exp((x-b2)/c2))


def f3(x, a3, b3, c3):
    return a3 / (1.0 + np.exp((x-b3)/c3))


def func(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):  # 논문에 나온 대로 logistic function 3개의 합
    return f1(x, a1, b1, c1) + f2(x, a2, b2, c2) + f3(x, a3, b3, c3) + d


def initial_params(yData):

    A = min(yData)
    B = max(yData)
    R = 0
    
    if np.mean(yData[:30]) > np.mean(yData[-30:]):
        inc = -1
    elif np.mean(yData[:30]) == np.mean(yData[-30:]):
        print("this is error please check")
        inc = 1
    else:
        inc = 1
        
    d = B
    a1 = 0.75 * (A - B)
    a2 = 0.125 * (A - B)
    a3 = 0.125 * (A - B)
    b1 = R
    b2 = R + 0.25  # paper +0.5
    b3 = R - 0.25  # paper -0.5
    c1 = inc * 0.15
    c2 = inc * 0.14
    c3 = inc * 0.14
    return [a1, b1, c1, a2, b2, c2, a3, b3, c3, d], A, B, inc

########################################################################################################################


@dataclass
class cSet:
    x: np.ndarray
    y: np.ndarray
    
    
@dataclass
class cMTF:
    x: np.ndarray
    y: np.ndarray
    MTF_at_Nyquist: float
    MTF_50: float
    

########################################################################################################################


class MTF:
    @staticmethod
    def GetESF(img, edgePoly):  # 점과 직선 사이의 거리
    
        if np.abs(edgePoly[0]) > 1000000:
    
            Y = img.shape[0]
            X = img.shape[1]
            values = np.reshape(img, X*Y)
            distance = np.zeros((Y, X))
            column = np.arange(0, X)
            for y in range(Y):
                distance[y, :] = (column - edgePoly[1])
            distances = np.reshape(distance, X*Y)
            temp_index = np.array(values > 0)
            values = values[temp_index]
            distances = distances[temp_index]
            indexes = np.argsort(distances)
            temp_len = indexes.size
            temp_middle = temp_len // 2
            temp_number = 30
            temp_small = max(0, temp_middle - temp_number)
            temp_large = temp_middle + temp_number
            sign = 1
            if np.average(values[indexes[temp_small:temp_middle]]) > np.average(values[indexes[temp_middle:temp_large]]):
                sign = -1
            values = values[indexes]
            distances = sign*distances[indexes]
            
            if distances[0] > distances[-1]:
                distances = np.flip(distances)
                values = np.flip(values)
            
            return cSet(distances, values)
        
        elif np.abs(edgePoly[0]) < 1000000:
            
            Y = img.shape[0]
            X = img.shape[1]
            values = np.reshape(img, X * Y)
            distance = np.zeros((Y, X))
            column = np.arange(0, X)
            for y in range(Y):
                distance[y, :] = \
                    (edgePoly[0] * column - y + edgePoly[1]) \
                    / np.sqrt(edgePoly[0] * edgePoly[0] + 1)
            distances = np.reshape(distance, X * Y)
            temp_index = np.array(values > 0)
            values = values[temp_index]
            distances = distances[temp_index]
            indexes = np.argsort(distances)
            temp_len = indexes.size
            temp_middle = temp_len // 2
            temp_number = 30
            temp_small = max(0, temp_middle - temp_number)
            temp_large = temp_middle + temp_number
            sign = 1
            if np.average(values[indexes[temp_small:temp_middle]]) > np.average(
                    values[indexes[temp_middle:temp_large]]):
                sign = -1
            values = values[indexes]
            distances = sign * distances[indexes]
    
            if distances[0] > distances[-1]:
                distances = np.flip(distances)
                values = np.flip(values)
    
            return cSet(distances, values)

    @staticmethod
    def GetUniformESF(img, img_orig, newedges_xy):
        
        edges_x = newedges_xy[:, 0]
        edges_y = newedges_xy[:, 1]
        
        np.std(edges_x)

        if np.std(edges_x) > 0.1 and np.std(edges_y) > 0.1:
            model = LinearRegression()
            model.fit(np.expand_dims(edges_x, -1), edges_y)
            edgePoly = np.array([model.coef_[0], model.intercept_])
            angle = math.degrees(math.atan(-edgePoly[0]))
            finalEdgePoly = edgePoly.copy()
            if angle > 0:
                img = np.flip(img, axis=1)
                img_orig = np.flip(img_orig, axis=1)
                finalEdgePoly[1] = np.polyval(edgePoly, np.size(img, 1) - 1)
                finalEdgePoly[0] = -edgePoly[0]
                edges_x = np.size(img, 1) - 1 - edges_x

            ESF = MTF.GetESF(img, finalEdgePoly)
            ESFValues = ESF.y
            ESFDistances = ESF.x
            
        elif np.std(edges_y) <= 0.1:
            edgePoly = np.array([0, np.mean(edges_y)])
            finalEdgePoly = edgePoly.copy()

            ESF = MTF.GetESF(img, finalEdgePoly)
            ESFValues = ESF.y
            ESFDistances = ESF.x

        else:
            edgePoly = np.array([1000000000, np.mean(edges_x)])
            finalEdgePoly = edgePoly.copy()

            ESF = MTF.GetESF(img, finalEdgePoly)
            ESFValues = ESF.y
            ESFDistances = ESF.x

        negative_signal = ESF.y[ESF.x < -5]
        positive_signal = ESF.y[ESF.x > +5]

        temp_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ESF_y_new = scipy.ndimage.convolve(ESFValues, temp_list, mode='nearest')/np.sum(temp_list)
        temp_inc = scipy.ndimage.convolve(ESF_y_new, (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1), mode='nearest')
        temp_inc = scipy.ndimage.convolve(temp_inc, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], mode='nearest')/11
        
        if np.argwhere((ESFDistances > 0) & (temp_inc <= 0)).size == 0:
            temp_inc_central_right = ESFDistances.size
        else:
            temp_inc_central_right = np.min(np.argwhere((ESFDistances > 0) & (temp_inc <= 0)))
        if np.argwhere((ESFDistances < 0) & (temp_inc <= 0)).size == 0:
            temp_inc_central_left = 0
        else:
            temp_inc_central_left = np.max(np.argwhere((ESFDistances < 0) & (temp_inc <= 0)))
        
        temp_inc_central_max = np.max(np.abs(temp_inc[temp_inc_central_left:temp_inc_central_right]))
        
        temp_inc = np.where(np.abs(temp_inc) < temp_inc_central_max*0.15, 0, temp_inc)

        ESF = cSet(ESFDistances[:], ESFValues[:])

        init_params, A, B, inc = initial_params(ESF.y)
        
        if inc == 1:
            bounds = (
                [1.0*(A-B), -0.25,      0, 0.25*(A-B),  0.0,      0, 0.25*(A-B), -0.25,      0, B-0.05],
                [0.5*(A-B),  0.25, np.inf,          0, 0.25, np.inf,          0,   0.0, np.inf, B+0.05]
            )
        else:
            bounds = (
                [1.0*(A-B), -0.25, -np.inf, 0.25*(A-B),  0.0, -np.inf, 0.25*(A-B), -0.25, -np.inf, B-0.05],
                [0.5*(A-B),  0.25,       0,          0, 0.25,       0,          0,   0.0,       0, B+0.05]
            )

        fittedParameters = curve_fit(
            func, ESF.x, ESF.y, p0=init_params,
            bounds=bounds,
            maxfev=100000
        )

        modelPredictions = func(ESF.x, *fittedParameters[0])
        
        absError = np.abs(modelPredictions - ESF.y)
        SE = np.square(absError)
        MSE = np.mean(SE)
        RMSE = np.sqrt(MSE)

        error_margin = 100.0
        error_margin_2 = 1.00  # 0.90, 0.95
        index_1 = np.where(absError > error_margin * RMSE)
        index_2 = np.where(absError > np.quantile(absError, error_margin_2))
        index = np.union1d(index_1[0], index_2[0])
        
        ESF_x_delete = np.delete(ESF.x, index)
        ESF_y_delete = np.delete(ESF.y, index)

        init_params2, A2, B2, inc2 = initial_params(ESF_y_delete)
        
        if inc2 == 1:
            bounds2 = (
                [1.0*(A2-B2), -0.25,      0, 0.25*(A2-B2),  0.0,      0, 0.25*(A2-B2), -0.25,      0, B2-0.05],
                [0.5*(A2-B2),  0.25, np.inf,            0, 0.25, np.inf,            0,   0.0, np.inf, B2+0.05]
            )
        else:
            bounds2 = (
                [1.0*(A2-B2), -0.25, -np.inf, 0.25*(A2-B2),  0.0, -np.inf, 0.25*(A2-B2), -0.25, -np.inf, B2-0.05],
                [0.5*(A2-B2),  0.25,       0,            0, 0.25,       0,            0,   0.0,       0, B2+0.05]
            )
        
        fittedParameters2 = curve_fit(
            func, ESF_x_delete, ESF_y_delete, p0=init_params2,
            bounds=bounds2,
            maxfev=100000
        )
        
        interval_num = 1000
        constant_margin_length = 5
        UniformDistances = np.linspace(ESF.x[0]-constant_margin_length, ESF.x[-1]+constant_margin_length, interval_num+1)
        UniformValues = func(UniformDistances, *fittedParameters2[0])
        temp_ESF_x = ESF.x
        temp_ESF_y = ESF.y
        ESF = cSet(UniformDistances, UniformValues)
        
        return img, img_orig, finalEdgePoly, ESF, ESFDistances, ESFValues, temp_inc, temp_ESF_x, temp_ESF_y, edges_x, negative_signal, positive_signal

    @staticmethod
    def GetLSF(ESF):  # Line Spread Function is differentiation of Edge Spread Function
        LSFnumerator = np.diff(ESF.y)
        LSFdenominator = np.diff(ESF.x)
        LSFValues = np.divide(LSFnumerator, LSFdenominator)
        LSFDistances = (ESF.x[0:-1]+ESF.x[1:])/2
        return cSet(LSFDistances, LSFValues)

    @staticmethod
    def GetMTF(LSF, multiplier):
    
        N = np.size(LSF.x)
        values = abs(fft(LSF.y) / abs(np.sum(LSF.y)))
        final_width = LSF.x[-1] - LSF.x[0]
        distances = np.arange(0, N) / final_width
        values = values[np.where(distances <= 1+1/final_width)]
        distances = distances[np.where(distances <= 1+1/final_width)]
    
        interpolate_num = 100
        interpolate_Distances = np.linspace(0, 1, interpolate_num+1)
    
        interpolate_func = interpolate.interp1d(distances, values, kind='cubic')
        invers_func = interpolate.interp1d(values, distances, kind='cubic')
        
        if values[-1] > 0.5:
            MTF_50 = 1.0
        else:
            MTF_50 = invers_func(0.5)
        interpolate_Values = interpolate_func(interpolate_Distances)
        MTF_at_Nyquist = interpolate_func(1.0*multiplier/2) * 100
        
        return cMTF(interpolate_Distances, interpolate_Values, MTF_at_Nyquist, MTF_50), distances, values

    @staticmethod
    def FigureMTF(img_orig, dir_temp, newedges_xy, data_name, data_dir, MAE, MSE, scale=1, verbose=False):
        img_orig = img_orig
        img = dir_temp
        
        img, img_orig, edgePoly, ESF, rawESF_x, rawESF_y, temp_inc, temp_ESF_x, temp_ESF_y, edges_x, negative_signal, positive_signal =\
            MTF.GetUniformESF(img, img_orig, newedges_xy)
        edges_y = newedges_xy[:, 1]
        
        if isinstance(img, int):
            return 0, 0
        
        # multiplier = 30 / img.shape[0]  # HR : 1, LR : 2
        multiplier = 1/scale
        
        xmin = ESF.x[0]
        xmax = ESF.x[-1]
        LSF = MTF.GetLSF(ESF)
        MTFinstance, distances, values = MTF.GetMTF(LSF, multiplier)

        if scale == 1:
            prefix = 'LR'
        else:
            prefix = 'SR'
        
        if verbose:
            plt.figure(figsize=(60, 50))
            
            plt.rc('font', size=75)
            plt.rc('axes', labelsize=75)
            plt.rc('xtick', labelsize=50)
            plt.rc('ytick', labelsize=50)
            plt.rc('legend', fontsize=50)
            plt.rc('figure', titlesize=50)
            
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('MTF Analysis')
            fig.tight_layout()
            
            gs1 = fig.add_gridspec(1, 3, left=0.05, right=1-0.05, bottom=0.55, top=0.95)
            gs2 = fig.add_gridspec(1, 1, left=0.05, right=1/2-0.05, bottom=0.05, top=0.45)
            gs3 = fig.add_gridspec(1, 1, left=1/2+0.05, right=1-0.05, bottom=0.05, top=0.45)
            ax1 = fig.add_subplot(gs2[0, 0])
            ax2 = fig.add_subplot(gs1[0, 0])
            ax3 = fig.add_subplot(gs1[0, 1])
            ax4 = fig.add_subplot(gs1[0, 2])
            ax5 = fig.add_subplot(gs3[0, 0])
            
            if np.std(edges_x) > 0.1 and np.std(edges_y) > 0.1:
                x = [0, np.size(img_orig, 1) - 1]
                y = np.polyval(edgePoly, x)
            elif np.std(edges_y) <= 0.1:
                x = [0, np.size(img, 1) - 1]
                y = [np.mean(edges_y), np.mean(edges_y)]
            else:
                x = [np.mean(edges_x), np.mean(edges_x)]
                y = [0, np.size(img, 0) - 1]
                
            ax1.plot(x, y, linewidth=10, alpha=0.50, color='red')
            ax1.imshow(img_orig, cmap='gray', vmin=0.0, vmax=1.0)
            ax1.scatter(edges_x, edges_y, s=100, c='white')
            ax1.set_xlim([-0.5, img.shape[1]-0.5])
            ax1.set_ylim([-0.5, img.shape[0]-0.5])
            np.rad2deg(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
            ax1.set_title(f"crop ROI")
            ax1.set_xlabel("px")
            ax1.set_ylabel("px")
            
            ax5.text(0.05, 0.9, f"name : {data_name}", size="medium")
            ax5.text(0.05, 0.8, f"edge degree : {np.round(math.degrees(math.atan(edgePoly[0])), 4)}°", size="medium")
            ax5.text(0.05, 0.7, f"ROI size : ({img_orig.shape[0]}, {img_orig.shape[1]})", size="medium")
            ax5.text(0.05, 0.6, f"SNR (Black) : {np.round(np.mean(negative_signal)/np.std(negative_signal), 4)}", size="medium")
            ax5.text(0.05, 0.5, f"SNR (White) : {np.round(np.mean(positive_signal)/np.std(positive_signal), 4)}", size="medium")
            ax5.text(0.05, 0.4, f"MAE : {np.round(MAE, 4)}", size="medium")
            ax5.text(0.05, 0.3, f"RMSE : {np.round(np.sqrt(MSE), 4)}", size="medium")
            ax5.text(0.05, 0.2, f"MTF : {np.round(MTFinstance.MTF_at_Nyquist,4)}%", size="medium")
            ax5.text(0.05, 0.1, f"evaluated time : {(datetime.datetime.now()).strftime('%Y%m%d_%H:%M:%S')}")
            ax5.set_xticks([])
            ax5.set_yticks([])
            ax5.grid(False)

            ax2.scatter(rawESF_x, rawESF_y, s=5, marker='x', color='orange', label=f'{rawESF_x.size} raw ESF')
            ax2.scatter(ESF.x, ESF.y, s=50, color='black', label=f'{ESF.x.size} uniform ESF')
            ax2.legend(prop={'size': 50}, loc=1)
            ax2.set_title(f"ESF")
            xticks = [xmin, xmax, 0]
            yticks = (np.arange(0, 1.05, 0.1).tolist())
            ax2.set_xticks(xticks)
            ax2.set_yticks(yticks)
            ax2.set_xlim([1.5*xmin, 1.5*xmax])
            ax2.set_ylim([0, 1])
            ax2.set_xlabel("px")
            ax2.grid(True)
    
            ax3.scatter(LSF.x, LSF.y, s=50, color='black', label=F'{LSF.x.size} uniform LSF')
            ax3.legend(prop={'size': 50}, loc=1)
            ax3.set_title("LSF")
            xticks = [xmin, xmax, 0]
            ax3.set_xticks(xticks)
            ax3.set_xlabel("px")
            ax3.grid(True)
            
            ax4.scatter(MTFinstance.x, MTFinstance.y, s=50, color='black', label=f'{MTFinstance.x.size} uniform MTF')
            ax4.scatter(distances, values, s=50, marker='x', color='orange', label='MTF')
            ax4.legend(prop={'size': 50}, loc=1)
            ax4.text(0.10, 1.075, "MTF$_{Nyq}$", ha="left", va="center", size="medium")
            ax4.text(0.333, 1.075, ":", ha="left", va="center", size="medium")
            ax4.text(0.35, 1.075, f" [{np.round(MTFinstance.MTF_at_Nyquist,3):.3f}]", ha="left", va="center", size="medium", color="red")
            ax4.text(0.65, 1.075, "% (LR nyq)", ha="left", va="center", size="medium")
            ax4.text(0.10, 1.025, "MTF$_{50}$", ha="left", va="center", size="medium")
            ax4.text(0.333, 1.025, ":", ha="left", va="center", size="medium")
            ax4.text(0.35, 1.025, f" [{np.round(MTFinstance.MTF_50 * scale,3):.3f}]", ha="left", va="center", size="medium", color="blue")
            ax4.text(0.65, 1.025, f"cycle/LR_pixel", ha="left", va="center", size="medium")
            ax4.set_xticks(np.linspace(0, 1, 11))
            ax4.set_yticks(np.linspace(0, 1, 11))
            ax4.vlines(x=1.0*multiplier/2, ymin=0, ymax=MTFinstance.MTF_at_Nyquist / 100., color='r', linewidth=5)
            ax4.hlines(y=MTFinstance.MTF_at_Nyquist/100., xmin=0, xmax=1.0*multiplier/2, color='r', linewidth=10)
            ax4.vlines(x=MTFinstance.MTF_50, ymin=0, ymax=0.5, color='b', linewidth=10)
            ax4.hlines(y=0.5, xmin=0, xmax=MTFinstance.MTF_50, color='b', linewidth=5)
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
            ax4.set_xlabel(f"cycle/{prefix}_px")
            ax4.grid(True)
    
            fig.tight_layout()
            plt.savefig(f"../results/MTF_{data_name}.png")
        
            plt.close()
        
        return MTFinstance.MTF_at_Nyquist, MTFinstance.MTF_50
