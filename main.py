import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import *
from scipy.io import wavfile
import matplotlib
import sys
import wave
from mainmenu import *
from player import *
from IIRfilter  import *
from FIRfilter import *
from PyQt5.QtWidgets import *
import os
import background
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
np.seterr(divide='ignore', invalid='ignore')
matplotlib.use('Qt5Agg')

# FIR类
class Firfilter(QMainWindow, FIRfilterWindow):
    def __init__(self, parent=None):
        super(Firfilter, self).__init__(parent)
        self.setupUi(self)
        self.mark = False
        self.flag = False
        self.fcut = None
        self.firType = None
        self.filterType = None
        self.wave_data = None
        self.time = None
        self.fft_signal_ = None
        self.Freq_ = None
        self.t = None
        self.yout = None
        self.FIR_b = None
        self.width = None
        self.fft_signal = None
        self.Freq = None
        self.playerFIR = None
        self.Kaiser_width = 0.85
        self.saveDatepath_FIR = os.getcwd().replace('\\', '/') + "/ProcessedSignal/text_fir.wav"
        self.numtaps = self.fs = self.f1 = self.f2 = self.f = None
        self.pushButton.clicked.connect(self.firdesign)
        self.pushButton_2.clicked.connect(self.firapply)
        self.pushButton_3.clicked.connect(self.compare)
        self.pushButton_4.clicked.connect(self.playback)
        self.pushButton_5.clicked.connect(self.zeropole)


    def firdesign(self):
        try:
            self.numtaps = int(self.lineEdit_5.text())
            self.fs = float(self.lineEdit.text())
            self.f1 = float(self.lineEdit_2.text())
            self.f2 = float(self.lineEdit_3.text())
            self.f = float(self.lineEdit_4.text())
            self.filterType = self.comboBox.currentText()
            self.firType = self.comboBox_2.currentText()
            if self.numtaps == ''or self.fs == ''or self.f1 == ''or self.f2 == ''or\
                    self.f == '' or self.filterType == '响应类型' or self.firType == '滤波器类型':
                QMessageBox.critical(self, '错误', '请输入完整滤波器参数!', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            else:
                if self.filterType == 'lowpass':
                    self.fcut = self.f * 2 / self.fs
                    if str(self.firType) == 'Kaiser':
                        self.width = self.Kaiser_width
                    self.FIR_b = signal.firwin(self.numtaps, self.fcut, width=self.width, window=str(self.firType))
                    # 绘制频率响应
                    wz, hz = signal.freqz(self.FIR_b)
                    plt.semilogx(wz * self.fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
                    plt.xlabel('Hz')
                    plt.ylabel('dB')
                    plt.title(str(self.firType))
                    plt.show()
                    self.mark = True
                    self.textBrowser.append('滤波器设计成功！')
                elif self.filterType == 'highpass':
                    self.fcut = self.f * 2 / self.fs
                    if str(self.firType) == 'Kaiser':
                        self.width = self.Kaiser_width
                    self.FIR_b = signal.firwin(self.numtaps, self.fcut, width=self.width, window=str(self.firType), pass_zero=False)
                    # 绘制频率响应
                    wz, hz = signal.freqz(self.FIR_b)
                    plt.semilogx(wz * self.fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
                    plt.xlabel('Hz')
                    plt.ylabel('dB')
                    plt.title(str(self.firType))
                    plt.show()
                    self.mark = True
                    self.textBrowser.append('滤波器设计成功！')
                elif self.filterType == 'bandpass':
                    self.fcut = [self.f1 * 2 / self.fs, self.f2 * 2 / self.fs]
                    if str(self.firType) == 'Kaiser':
                        self.width = self.Kaiser_width
                    self.FIR_b = signal.firwin(self.numtaps, self.fcut, width=self.width, window=str(self.firType), pass_zero=False)
                    # 绘制频率响应
                    wz, hz = signal.freqz(self.FIR_b)
                    plt.semilogx(wz * self.fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
                    plt.xlabel('Hz')
                    plt.ylabel('dB')
                    plt.title(str(self.firType))
                    plt.show()
                    self.mark = True
                    self.textBrowser.append('滤波器设计成功！')
                elif self.filterType == 'bandstop':
                    self.fcut = [self.f1 * 2 / self.fs, self.f2 * 2 / self.fs]
                    if str(self.firType) == 'Kaiser':
                        self.width = self.Kaiser_width
                    self.FIR_b = signal.firwin(self.numtaps, self.fcut, width=self.width, window=str(self.firType))
                    # 绘制频率响应
                    wz, hz = signal.freqz(self.FIR_b)
                    plt.semilogx(wz * self.fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
                    plt.xlabel('Hz')
                    plt.ylabel('dB')
                    plt.title(str(self.firType))
                    plt.show()
                    self.mark = True
                    self.textBrowser.append('滤波器设计成功！')
        except Exception as e:
            print(e)

    def firapply(self):
        if self.mark:
            try:
                f = wave.open(filename, 'rb')  # 打开音频
                params = f.getparams()  # 读取格式化信息
                nchannels, sampwidth, framerate, nframes = params[:4]  # （声道数，量化位数，采样频率，采样点数），注：wave模块支支持非压缩数据
                str_data = f.readframes(nframes)  # 读取波形数据
                f.close()
                self.wave_data = np.frombuffer(str_data, dtype=np.short)  # 将字符串转化为一维short类型数组
                if nchannels == 1:  # 单声道
                    self.wave_data.shape = -1, 1
                    maximum = max(abs(self.wave_data))
                    self.wave_data = self.wave_data * 1.0 / maximum
                    self.time = np.arange(0, nframes) * (1.0 / framerate)
                    # 滤波
                    self.t = np.linspace(0, nframes / framerate, nframes, endpoint=False)
                    self.yout = signal.filtfilt(self.FIR_b, 1, self.wave_data[:, 0])
                    # 滤波后进行FFT
                    self.fft_signal_ = np.fft.fft(self.yout)
                    self.fft_signal_ = np.fft.fftshift(abs(self.fft_signal_))
                    self.fft_signal_ = self.fft_signal_[int(self.fft_signal_.shape[0] / 2):]
                    # 建立频率轴
                    self.Freq_ = np.arange(0, framerate / 2, framerate / (2 * len(self.fft_signal_)))
                    # 原声进行FFT 频域
                    sampling_freq, audio = wavfile.read(filename)
                    audio = audio * 1.0 / (max(abs(audio)))
                    self.fft_signal = np.fft.fft(audio)
                    self.fft_signal = abs(self.fft_signal)
                    self.fft_signal = np.fft.fftshift(self.fft_signal)
                    self.fft_signal = self.fft_signal[int(self.fft_signal.shape[0] / 2):]
                    self.Freq = np.arange(0, sampling_freq / 2, sampling_freq / (2 * len(self.fft_signal)))
                    # 写音频文件
                    self.yout = self.yout * maximum  # 去归一化
                    self.yout = self.yout.astype(np.short)
                    os.remove(self.saveDatepath_FIR)
                    f = wave.open(self.saveDatepath_FIR, "wb")
                    f.setnchannels(nchannels)
                    f.setsampwidth(sampwidth)
                    f.setframerate(framerate)
                    f.setnframes(nframes)
                    f.writeframes(self.yout.tobytes())
                    f.close()
                    self.flag = True
                    self.textBrowser.append('滤波成功!')
                else:
                    self.textBrowser.append("本系统目前只支持单声道")
            except Exception as e:
                print(e)
        else:
            self.textBrowser.append('未检测到滤波器，请先设计滤波器!')

    def compare(self):
        try:
            if self.mark:
                if self.flag:
                    plt.subplot(221)
                    plt.plot(self.time, self.wave_data[:, 0])
                    plt.xlabel('时间/s', fontsize=14)
                    plt.ylabel('幅度', fontsize=14)
                    plt.xlabel('原声时域波形')
                    plt.subplot(222)
                    plt.plot(self.Freq, self.fft_signal)
                    plt.xlabel('原声频域波形')
                    plt.subplot(223)
                    plt.plot(self.t, self.yout)
                    plt.xlabel('时间')
                    plt.ylabel('幅度')
                    plt.xlabel('滤波后时域波形')
                    plt.subplot(224)
                    plt.plot(self.Freq_, self.fft_signal_)
                    plt.xlabel('频率')
                    plt.ylabel('幅值')
                    plt.xlabel('滤波后频域波形')
                    plt.tight_layout()
                    plt.show()
                else:
                    self.textBrowser.append('请先应用滤波器对音频进行滤波!')
            else:
                self.textBrowser.append('未检测到滤波器，请先设计滤波器!')
        except Exception as e:
            print(e)

    def playback(self):
        if self.flag:

            self.playerFIR = QMediaPlayer(self)

            self.playerFIR.pause()

            self.playerFIR.setMedia(QMediaContent(QUrl(self.saveDatepath_FIR)))
            self.playerFIR.play()
        else:
            self.textBrowser.append('请先应用滤波器对音频进行滤波!')

    def zeropole(self):      # 如果传点长度过大，会报错，因为指数函数太大了
        try:
            if self.mark:
                fir_a = np.zeros(self.numtaps)
                fir_a[self.numtaps - 1] = 1
                z1, p1, k1 = signal.tf2zpk(self.FIR_b, fir_a)  # zero, pole and gain
                c = np.vstack((fir_a, self.FIR_b))
                Max = (abs(c)).max()  # find the largest value
                a = fir_a / Max  # normalization
                b = self.FIR_b / Max
                Ra = (a * (2 ** ((self.numtaps - 1) - 1))).astype(int)  # quantizan and truncate
                Rb = (b * (2 ** ((self.numtaps - 1) - 1))).astype(int)
                z2, p2, k2 = signal.tf2zpk(Rb, Ra)
                # 参数方程画圆
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = np.cos(theta)
                y = np.sin(theta)
                plt.plot(x, y, color='black')
                for i in p1:
                    plt.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
                for i in z1:
                    plt.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
                for i in p2:
                    plt.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
                for i in z2:
                    plt.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
                plt.xlim(-1.8, 1.8)
                plt.ylim(-1.2, 1.2)
                plt.grid()
                plt.title("{0} bit quantization".format(self.numtaps))
                plt.show()
            else:
                self.textBrowser.append('未检测到滤波器，请先设计滤波器!')
        except Exception as e:
            print(e)

# IIR类
class IIrfilter(QMainWindow, IIRfilterWindow):
    def __init__(self, parent=None):
        super(IIrfilter, self).__init__(parent)
        self.setupUi(self)
        self.saveDatepath_IIR = os.getcwd().replace('\\', '/') + "/ProcessedSignal/text_iir.wav"
        self.t = None
        self.playerIIR = None
        self.fs = None
        self.flag = None
        self.wp = None
        self.fft_signal = None
        self.fft_signal_ = None
        self.Freq = None
        self.Freq_ = None
        self.time = None
        self.wave_data = None
        self.yout = None
        self.ws = None
        self.Rp = None
        self.As = None
        self.filterType = None
        self.iirType = None
        self.filtz = self.filts = None
        self.N = None
        self.Wn = None
        self.z = None
        self.N = None
        self.p = None
        self.mark = False
        self.pushButton.clicked.connect(self.iirdesign)
        self.pushButton_2.clicked.connect(self.iirapply)
        self.pushButton_3.clicked.connect(self.compare)
        self.pushButton_4.clicked.connect(self.playback)
        self.pushButton_5.clicked.connect(self.zeropole)
        self.pushButton_6.clicked.connect(self.groupdelay)

    def iirdesign(self):
        try:
            self.fs = self.lineEdit.text()
            self.wp = self.lineEdit_2.text()
            self.ws = self.lineEdit_3.text()
            self.Rp = self.lineEdit_4.text()
            self.As = self.lineEdit_5.text()
            self.filterType = self.comboBox.currentText()
            self.iirType = self.comboBox_2.currentText()
            if self.fs == '' or self.wp == '' or self.ws == '' or self.Rp == '' or self.As == '' or \
                    self.filterType == '响应类型' or self.iirType == '滤波器类型':
                QMessageBox.critical(self, '错误', '请输入完整滤波器参数!', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            else:
                if str(self.iirType) == 'butter':
                    fs = float(self.fs)
                    if str(self.filterType) == 'bandpass' or str(self.filterType) == 'bandstop':
                        wp = str(self.wp).split()  # 巴特沃斯需输入两个通带截止频率和两个阻带截止频率
                        wp0 = float(wp[0]) * (2 * np.pi / fs)
                        wp1 = float(wp[1]) * (2 * np.pi / fs)
                        wp[0] = (2 * fs) * np.tan(wp0 / 2)  # 双线性变换
                        wp[1] = (2 * fs) * np.tan(wp1 / 2)
                        omiga_p = [float(wp[0]), float(wp[1])]      # 频率预畸
                        ws = str(self.ws).split()  # 切分开之后再转换为array
                        wst0 = float(ws[0]) * (2 * np.pi / fs)
                        wst1 = float(ws[1]) * (2 * np.pi / fs)
                        ws[0] = (2 * fs) * np.tan(wst0 / 2)  # 双线性变换
                        ws[1] = (2 * fs) * np.tan(wst1 / 2)
                        omiga_st = [ws[0], ws[1]]           # 频率预畸
                    else:
                        wp = float(self.wp) * (2 * np.pi / fs)
                        ws = float(self.ws) * (2 * np.pi / fs)
                        omiga_p = (2 * fs) * np.tan(wp / 2)    # 频率预畸
                        omiga_st = (2 * fs) * np.tan(ws / 2)       # 频率预畸
                    self.Rp = float(self.Rp)
                    self.As = float(self.As)
                    self.N, self.Wn = signal.buttord(omiga_p, omiga_st, self.Rp, self.As, True)
                    self.filts = signal.lti(*signal.butter(self.N, self.Wn, btype=str(self.filterType), analog=True))
                    self.filtz = signal.lti(*signal.bilinear(self.filts.num, self.filts.den, fs))
                    self.z, self.p = signal.bilinear(self.filts.num, self.filts.den, fs)
                    wz, hz = signal.freqz(self.filtz.num, self.filtz.den)
                    plt.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
                    plt.xlabel('Hz')
                    plt.ylabel('dB')
                    plt.title('巴特沃斯')
                    plt.show()
                    self.mark = True
                    self.textBrowser.append('滤波器设计成功！')

                elif str(self.iirType) == 'chebyI':  # 切比雪夫一型
                    fs = float(self.fs)
                    if str(self.filterType) == 'bandpass' or str(self.filterType) == 'bandstop':
                        wp = str(self.wp).split()  # 切分开之后再转换为array
                        wp0 = float(wp[0]) * (2 * np.pi / fs)
                        wp1 = float(wp[1]) * (2 * np.pi / fs)
                        wp[0] = (2 * fs) * np.tan(wp0 / 2)   # 双线性变换
                        wp[1] = (2 * fs) * np.tan(wp1 / 2)
                        omiga_p = [float(wp[0]), float(wp[1])]
                        ws = str(self.ws).split()    # 切分开之后再转换为array
                        wst0 = float(ws[0]) * (2 * np.pi / fs)
                        wst1 = float(ws[1]) * (2 * np.pi / fs)
                        ws[0] = (2 * fs) * np.tan(wst0 / 2)  # 双线性变换
                        ws[1] = (2 * fs) * np.tan(wst1 / 2)
                        omiga_st = [ws[0], ws[1]]
                    else:
                        wp = float(self.wp) * (2 * np.pi / fs)
                        ws = float(self.ws) * (2 * np.pi / fs)
                        omiga_p = (2 * fs) * np.tan(wp / 2)
                        omiga_st = (2 * fs) * np.tan(ws / 2)

                    if len(str(self.Rp).split()) > 1:   # 纹波参数
                        Rpinput = str(self.Rp).split()
                        self.Rp = float(Rpinput[0])
                        self.As = float(self.As)
                        rp_in = float(Rpinput[1])
                    else:
                        self.Rp = float(self.Rp)
                        self.As = float(self.As)
                        rp_in = 0.1 * self.Rp
                    self.N, self.Wn = signal.cheb1ord(omiga_p, omiga_st, self.Rp, self.As, True)
                    self.filts = signal.lti(*signal.cheby1(self.N, rp_in, self.Wn, btype=str(self.filterType), analog=True))  # 切比雪夫有纹波参数
                    self.filtz = signal.lti(*signal.bilinear(self.filts.num, self.filts.den, fs))
                    self.z, self.p = signal.bilinear(self.filts.num, self.filts.den, fs)
                    wz, hz = signal.freqz(self.filtz.num, self.filtz.den)
                    plt.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
                    plt.title('切比雪夫1型')
                    plt.show()
                    self.mark = True
                    self.textBrowser.append('滤波器设计成功！')

                elif str(self.iirType) == 'chebyII':  # 切比雪夫二型
                    fs = float(self.fs)
                    if str(self.filterType) == "bandpass" or str(self.filterType) == "bandstop":
                        wp = str(self.wp).split()   # 切分开之后再转换为array
                        wp0 = float(wp[0]) * (2 * np.pi / fs)
                        wp1 = float(wp[1]) * (2 * np.pi / fs)
                        wp[0] = (2 * fs) * np.tan(wp0 / 2)  # 双线性变换
                        wp[1] = (2 * fs) * np.tan(wp1 / 2)
                        omiga_p = [float(wp[0]), float(wp[1])]
                        ws = str(self.ws).split()   # 切分开之后再转换为array
                        wst0 = float(ws[0]) * (2 * np.pi / fs)
                        wst1 = float(ws[1]) * (2 * np.pi / fs)
                        ws[0] = (2 * fs) * np.tan(wst0 / 2)   # 双线性变换
                        ws[1] = (2 * fs) * np.tan(wst1 / 2)
                        omiga_st = [ws[0], ws[1]]
                    else:
                        wp = float(self.wp) * (2 * np.pi / fs)
                        ws = float(self.ws) * (2 * np.pi / fs)
                        omiga_p = (2 * fs) * np.tan(wp / 2)
                        omiga_st = (2 * fs) * np.tan(ws / 2)

                    if len(str(self.As).split()) > 1:  # 纹波参数
                        Asinput = str(self.As).split()
                        self.As = float(Asinput[0])
                        self.Rp = float(self.Rp)
                        rs_in = float(Asinput[1])
                    else:
                        self.Rp = float(self.Rp)
                        self.As = float(self.As)
                        rs_in = 0.1 * self.As
                    self.N, self.Wn = signal.cheb2ord(omiga_p, omiga_st, self.Rp, self.As, True)
                    self.filts = signal.lti(*signal.cheby2(self.N, rs_in, self.Wn, btype=str(self.filterType), analog=True))
                    self.filtz = signal.lti(*signal.bilinear(self.filts.num, self.filts.den, fs))
                    self.z, self.p = signal.bilinear(self.filts.num, self.filts.den, fs)
                    wz, hz = signal.freqz(self.filtz.num, self.filtz.den)
                    plt.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
                    plt.title('切比雪夫II型')
                    plt.show()
                    self.mark = True
                    self.textBrowser.append('滤波器设计成功！')

                elif str(self.iirType) == 'ellip':  # 椭圆函数
                    fs = float(self.fs)
                    if str(self.filterType) == "bandpass" or str(self.filterType) == "bandstop":
                        wp = str(self.wp).split()    # 切分开之后再转换为array
                        wp0 = float(wp[0]) * (2 * np.pi / fs)
                        wp1 = float(wp[1]) * (2 * np.pi / fs)
                        wp[0] = (2 * fs) * np.tan(wp0 / 2)   # 双线性变换
                        wp[1] = (2 * fs) * np.tan(wp1 / 2)
                        omiga_p = [float(wp[0]), float(wp[1])]
                        ws = str(self.ws).split()    # 切分开之后再转换为array
                        wst0 = float(ws[0]) * (2 * np.pi / fs)
                        wst1 = float(ws[1]) * (2 * np.pi / fs)
                        ws[0] = (2 * fs) * np.tan(wst0 / 2)     # 双线性变换
                        ws[1] = (2 * fs) * np.tan(wst1 / 2)
                        omiga_st = [ws[0], ws[1]]
                    else:
                        wp = float(self.wp) * (2 * np.pi / fs)
                        ws = float(self.ws) * (2 * np.pi / fs)
                        omiga_p = (2 * fs) * np.tan(wp / 2)
                        omiga_st = (2 * fs) * np.tan(ws / 2)
                    if len(str(self.As).split()) > 1:
                        Asinput = str(self.As).split()
                        self.As = float(Asinput[0])
                        rs_in = float(Asinput[1])
                        if len(str(self.Rp).split()) > 1:
                            Rpinput = str(self.Rp).split()
                            self.Rp = float(Rpinput[0])
                            rp_in = float(Rpinput[1])
                        else:
                            self.Rp = float(self.Rp)
                            rp_in = 0.1 * self.Rp
                    else:
                        self.As = float(self.As)
                        rs_in = 0.1 * self.As
                        if len(str(self.Rp).split()) > 1:
                            Rpinput = str(self.Rp).split()
                            self.Rp = float(Rpinput[0])
                            rp_in = float(Rpinput[1])
                        else:
                            self.Rp = float(self.Rp)
                            rp_in = 0.1 * self.Rp
                    self.N, self.Wn = signal.ellipord(omiga_p, omiga_st, self.Rp, self.As, True)
                    self.filts = signal.lti(*signal.ellip(self.N, rp_in, rs_in, self.Wn, btype=str(self.filterType), analog=True))
                    self.filtz = signal.lti(*signal.bilinear(self.filts.num, self.filts.den, fs))
                    self.z, self.p = signal.bilinear(self.filts.num, self.filts.den, fs)
                    wz, hz = signal.freqz(self.filtz.num, self.filtz.den)
                    plt.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
                    plt.title('椭圆函数')
                    plt.show()
                    self.mark = True
                    self.textBrowser.append('滤波器设计成功！')
        except Exception as e:
            print(e)

    def iirapply(self):
        if self.mark:
            try:
                f = wave.open(filename, 'rb')  # 打开音频
                params = f.getparams()  # 读取格式化信息
                nchannels, sampwidth, framerate, nframes = params[:4]  # （声道数，量化位数，采样频率，采样点数），注：wave模块支支持非压缩数据
                str_data = f.readframes(nframes)  # 读取波形数据
                f.close()
                self.wave_data = np.frombuffer(str_data, dtype=np.short)  # 将字符串转化为一维short类型数组
                if nchannels == 1:  # 单声道
                    self.wave_data.shape = -1, 1
                    maximum = max(abs(self.wave_data))
                    self.wave_data = self.wave_data * 1.0 / maximum
                    self.time = np.arange(0, nframes) * (1.0 / framerate)
                    # 滤波
                    self.t = np.linspace(0, nframes / framerate, nframes, endpoint=False)
                    self.yout = signal.filtfilt(self.z, self.p, self.wave_data[:, 0], method='gust')
                    # 滤波后进行FFT
                    self.fft_signal_ = np.fft.fft(self.yout)
                    self.fft_signal_ = np.fft.fftshift(abs(self.fft_signal_))
                    self.fft_signal_ = self.fft_signal_[int(self.fft_signal_.shape[0] / 2):]
                    # 建立频率轴
                    self.Freq_ = np.arange(0, framerate / 2, framerate / (2 * len(self.fft_signal_)))
                    # 原声进行FFT 频域
                    sampling_freq, audio = wavfile.read(filename)
                    audio = audio * 1.0 / (max(abs(audio)))
                    self.fft_signal = np.fft.fft(audio)
                    self.fft_signal = abs(self.fft_signal)
                    self.fft_signal = np.fft.fftshift(self.fft_signal)
                    self.fft_signal = self.fft_signal[int(self.fft_signal.shape[0] / 2):]
                    self.Freq = np.arange(0, sampling_freq / 2, sampling_freq / (2 * len(self.fft_signal)))
                    # 写音频文件
                    self.yout = self.yout * maximum  # 去归一化
                    self.yout = self.yout.astype(np.short)
                    os.remove(self.saveDatepath_IIR)
                    f = wave.open(self.saveDatepath_IIR, "wb")
                    f.setnchannels(nchannels)
                    f.setsampwidth(sampwidth)
                    f.setframerate(framerate)
                    f.setnframes(nframes)
                    f.writeframes(self.yout.tobytes())
                    f.close()
                    self.flag = True
                    self.textBrowser.append('滤波成功!')
                else:
                    self.textBrowser.append('本系统目前只用于处理单声道!')
            except Exception as e:
                print(e)
        else:
            self.textBrowser.append('未检测到滤波器，请先设计滤波器!')

    def compare(self):
        try:
            if self.mark:
                if self.flag:
                    plt.subplot(221)
                    plt.plot(self.time, self.wave_data[:, 0])
                    plt.xlabel('时间/s', fontsize=14)
                    plt.ylabel('幅度', fontsize=14)
                    plt.xlabel('原声时域波形')
                    plt.subplot(222)
                    plt.plot(self.Freq, self.fft_signal)
                    plt.xlabel('原声频域波形')
                    plt.subplot(223)
                    plt.plot(self.t, self.yout)
                    plt.xlabel('时间')
                    plt.ylabel('幅度')
                    plt.xlabel('滤波后时域波形')
                    plt.subplot(224)
                    plt.plot(self.Freq_, self.fft_signal_)
                    plt.xlabel('频率')
                    plt.ylabel('幅值')
                    plt.xlabel('滤波后频域波形')
                    plt.tight_layout()
                    plt.show()
                else:
                    self.textBrowser.append('请先应用滤波器对音频进行滤波!')
            else:
                self.textBrowser.append('未检测到滤波器，请先设计滤波器!')
        except Exception as e:
            print(e)

    def playback(self):
        if self.flag:
            self.playerIIR = QMediaPlayer(self)
            self.playerIIR.pause()
            self.playerIIR.setMedia(QMediaContent(QUrl(self.saveDatepath_IIR)))
            self.playerIIR.play()
        else:
            self.textBrowser.append('请先应用滤波器对音频进行滤波!')

    def zeropole(self):       # 零极点分布
        try:
            if self.mark:
                N = self.N
                z1, p1, k1 = signal.tf2zpk(self.z, self.p)  # zero, pole and gain
                c = np.vstack((self.p, self.z))
                Max = (abs(c)).max()  # find the largest value
                a = self.p / Max  # normalization
                b = self.z / Max
                Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # quantizan and truncate
                Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
                z2, p2, k2 = signal.tf2zpk(Rb, Ra)
                # 参数方程画圆
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = np.cos(theta)
                y = np.sin(theta)
                plt.plot(x, y, color='black')
                for i in p1:
                    plt.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
                for i in z1:
                    plt.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
                for i in p2:
                    plt.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
                for i in z2:
                    plt.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
                plt.xlim(-1.8, 1.8)
                plt.ylim(-1.2, 1.2)
                plt.grid()
                plt.title("{0} bit quantization".format(N))
                plt.show()
            else:
                self.textBrowser.append('未检测到滤波器，请先设计滤波器!')
        except Exception as e:
            print(e)

    def groupdelay(self):
        try:
            if self.mark:
                w1, gd = signal.group_delay((self.filtz.num, self.filtz.den))
                plt.figure()
                plt.plot(w1, gd)
                plt.title('群延迟')
                plt.show()
            else:
                self.textBrowser.append('未检测到滤波器，请先设计滤波器!')
        except Exception as e:
            print(e)


# 主菜单设计
class Design(QMainWindow, MainmenuWindow):
    def __init__(self, parent=None):
        self.path = None
        super(Design, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.playwindow)
        self.pushButton_2.clicked.connect(self.analysis)
        self.pushButton_3.clicked.connect(self.irrfilterwin)
        self.pushButton_4.clicked.connect(self.firfilterwin)
        self.actionOpen_compile.triggered.connect(self.openpagefile)
        self.actionExit.triggered.connect(self.close)

    @staticmethod
    def playwindow():
        ui_xiling.show()

    def openpagefile(self):
        self.path, _ = QFileDialog.getOpenFileName(self, '打开文件', '', '音乐文件 (*.wav)')
        global filename
        filename = self.path

    # 画出噪声干扰的语音信号时域波形；然后画出语音信号的功率谱波形
    def analysis(self):
        if filename:
            f = wave.open(filename, 'rb')       # 打开音频
            params = f.getparams()              # 读取格式化信息
            nchannels, sampwidth, framerate, nframes = params[:4]  # （声道数，量化位数，采样频率，采样点数），注：wave模块支支持非压缩数据
            str_data = f.readframes(nframes)   # 读取波形数据
            f.close()
            wave_data = np.frombuffer(str_data, dtype=np.short)  # 将字符串转化为一维short类型数组
            if nchannels == 1:      # 单声道
                wave_data.shape = -1, 1
                wave_data = wave_data.T
                time = np.arange(0, nframes)*(1.0/framerate)
                plt.subplot(211)
                plt.plot(time, wave_data[0])
                plt.xlabel('时间/s', fontsize=14)
                plt.ylabel('幅度', fontsize=14)
                plt.xlabel('时域波形')
                sampling_freq, audio = wavfile.read(filename)
                audio = audio * 1.0 / (max(abs(audio)))
                fft_signal = np.fft.fft(audio)
                fft_signal = abs(fft_signal)
                fft_signal = np.fft.fftshift(fft_signal)
                fft_signal = fft_signal[int(fft_signal.shape[0] / 2):]
                # freqInteral = (sampling_freq / len(fft_signal))
                Freq = np.arange(0, sampling_freq / 2, sampling_freq / (2 * len(fft_signal)))
                plt.subplot(212)
                plt.plot(Freq, fft_signal)
                plt.show()

        else:
            QMessageBox.critical(self, '错误', '请先打开一个音频!', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    @staticmethod
    def irrfilterwin():
        ui_xiling2.show()

    @staticmethod
    def firfilterwin():
        ui_xiling3.show()


# 播放音频
class Play(PlayerWindow, QMainWindow):
    def __init__(self, parent=None):
        self.player = None
        super(Play, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.playwav)


    def playwav(self):
        if filename:
            self.player = QMediaPlayer(self)
            self.player.pause()
            self.player.setMedia(QMediaContent(QUrl(filename)))
            self.player.play()
            self.textBrowser.append('播放成功')
            f = wave.open(filename, 'rb')  # 打开音频
            params = f.getparams()  # 读取格式化信息
            nchannels, sampwidth, framerate, nframes = params[:4]   # 声道数，量化位数，采样频率，采样点数
            self.textBrowser.append('声道数：{0}\n量化位数：{1}\n采样频率：{2}\n采样点数：{3}'.format(nchannels, sampwidth, framerate, nframes))
        else:
            self.textBrowser.append('未找到音频')




if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = Design()
    ui_xiling   = Play()
    ui_xiling2 = IIrfilter()
    ui_xiling3 = Firfilter()
    filename = ''
    myWin.show()
    sys.exit(app.exec_())