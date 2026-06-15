# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:47:28 2020
增加RF观测信号的不确定性，包括虚警率、多径概率，虚警率，
@author: Administrator
生成目标的距离增强，并调整雷达杂波的生成方式
"""

#机动目标3维运动仿真

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import scipy.io as scio
import copy

#根据文献111：
g = 10  # 重力加速度，单位m/s^2
#载荷转速度变化率, 角度都是弧度
def Load2Acceleration(nz, nx, Phi, velocity, theta):
    velocity_a = g * (nx - np.sin(theta))  # 式1
    #为了避免奇点：
    if velocity == 0:
        #假如没有速度，则认为飞机静止，而不是无穷快地旋转
        theta_a = 0
        psi_a = 0
    else:
        theta_a = g * (nz * np.cos(Phi) - np.cos(theta)) / velocity  # 式2
        if np.cos(theta) == 0:
            #假如是垂直的速度，则把方位角设定为0，而不是无穷快地自转
            psi_a = 0
        else:
            psi_a = g * nz * np.sin(Phi) / (velocity * np.cos(theta))  # 式3
    #去掉角度周期性
    theta_a = theta_a % (np.pi*2)
    psi_a = psi_a % (np.pi*2)
    if theta_a > np.pi:
        theta_a -= np.pi*2
    if psi_a > np.pi:
        psi_a -= np.pi*2
    return velocity_a, theta_a, psi_a
#速度变化率转载荷, 角度都是弧度
def Acceleration2Load(velocity_a, theta_a, psi_a, velocity, theta):
    nx = velocity_a / g + np.sin(theta)
    # 定义： X = nz * np.cos(Phi)                            ----式7
    # 定义： Y = nz * np.sin(Phi)                            ----式8
    X = theta_a * velocity / g + np.cos(theta)
    Y = psi_a * velocity * np.cos(theta) / g
    # self.nz = np.sqrt(np.square(X) + np.square(Y))
    Phi = np.arctan2(Y, X)
    if np.cos(Phi) == 0:
        nz = Y / np.sin(Phi)
    else:
        nz = X / np.cos(Phi)
    return nz, nx, Phi

def normal_distribution_truncation(min_value, max_value, n_sigma):
    # n_sigma 表示几个正太分布的n sigma原则
    mean = (min_value + max_value)/2
    std = (max_value - min_value)/(n_sigma * 2)
    r = np.random.normal(mean, std, 1)
    while (r<min_value or r>max_value):
        r = np.random.normal(mean, std, 1)
    return r

#机动目标状态更新，目标状态更新的误差通过载荷引入，因为载荷与力直接相关。
class maneuvering_target_state_update(object):
    def __init__(self, sT=0.1, xc=1000, yc=1000, zc=1000, velocity=60, theta=30, psi=30, nz=0.1, nx=0.1, Phi=10,
                 anz=0.01, anx=0.01, aPhi=0.01,  sgz=0.1, sgx=0.1, sgp=1):
        #输入角度的单位都是角度（为了好看），类里的角度变量全部转换成弧度
        # Sampling interval (s)
        self.sT = sT

        #机动目标初始状态设定----------------------------------------------------------------------------------------------
        # 起始空间位置设定
        self.xc = xc  # 单位m
        self.yc = yc  # 单位m
        self.zc = zc  # 单位m
        # 起始速度设定，通过速度的分解，可以计算速度在笛卡尔坐标系的分量
        self.velocity = velocity  # 单位m/s
        self.theta = theta * np.pi / 180  # 单位°转rad
        self.psi = psi * np.pi / 180  # 单位°转rad

        #载荷的衰减控制
        self.anz = anz  # 变化率，无单位
        self.anx = anx  # 变化率，无单位
        self.aPhi = aPhi  # 变化率，无单位
        # 载荷的噪声的设定, 标准差
        self.sgz = sgz  # 运动模型中nz的标准差，无单位
        self.sgx = sgx  # 运动模型中nx的标准差，无单位
        self.sgp = sgp * np.pi / 180  # 运动模型中phi的标准差，角度，单位°

        self.nz = nz  # 正常载荷，无单位
        self.nx = nx # 切向载荷，无单位
        self.Phi = Phi * np.pi / 180  # roll angle，单位rad

    def State_update_by_load_control(self):
        #目标下一个状态的计算,通过载荷控制
        #提前控制，先改控制量再更新
        #控制量加噪
        self.nz += np.random.normal(0, self.sgz)  # 正常载荷，无单位
        self.nx += np.random.normal(0, self.sgx)  # 切向载荷，无单位
        self.Phi += np.random.normal(0, self.sgp)  # roll angle，单位rad

        # 目标速度变化量的更新，velocity_a，theta_a, psi_a
        self.velocity_a, self.theta_a, self.psi_a = Load2Acceleration(self.nz, self.nx, self.Phi, self.velocity, self.theta)
        #目标位置的更新，x, y, z
        self.xc += self.velocity * self.sT * np.cos(self.theta) * np.cos(self.psi)
        self.yc += self.velocity * self.sT * np.cos(self.theta) * np.sin(self.psi)
        self.zc += self.velocity * self.sT * np.sin(self.theta)

        # 目标速度的更新，velocity，theta, psi
        self.velocity += self.velocity_a * self.sT
        self.theta += self.theta_a * self.sT
        self.psi += self.psi_a * self.sT

    def Load_control_update(self, nz, nx, Phi):
        self.nz = nz
        self.nx = nx
        self.Phi = Phi

class Trajectory_auto_generator_one_target(object):
    # def __init__(self, sT=0.1, x_min=-500, x_max=500, y_min=-500, y_max=500, z_min= 300, z_max= 500,
    #              v_min=2, v_max=60, theta_min= -90, theta_max= 90, psi_min= -180, psi_max= 180, mode=1,
    #              a_min=-5, a_max=5, b_min= 0*np.pi/180, b_max= 0*np.pi/180, c_min=-10*np.pi/180, c_max=10*np.pi/180,
    #              SectionLen=20, P_maneuver=0.2, P_dead=0.08, MaxLastLen=2000):

    def __init__(self, sT=0.1, r_min=100, r_max=5000, az_min=-180, az_max=180, el_min=3, el_max=60,
                 v_min=2, v_max=60, theta_min=-90, theta_max=90, psi_min=-180, psi_max=180, mode=1,
                 a_min=-5, a_max=5, b_min=0 * np.pi / 180, b_max=0 * np.pi / 180, c_min=-10 * np.pi / 180,
                 c_max=10 * np.pi / 180,
                 SectionLen=20, P_maneuver=0.2, P_dead=0.08, MaxLastLen=2000):
        #目标状态生成的范围：
        # # 航迹起始点范围：主要考虑高度，水平面就大概给个范围：单位（m）
        # self.x_min = x_min
        # self.x_max = x_max
        # self.y_min = y_min
        # self.y_max = y_max
        # self.z_min = z_min
        # self.z_max = z_max
        # xc = np.random.uniform(self.x_min, self.x_max)
        # yc = np.random.uniform(self.y_min, self.y_max)
        # zc = np.random.uniform(self.z_min, self.z_max)

        # 速度范围：考虑无人机能飞的最快速度以及模型中速度低于2会出现的问题，单位（m/s）,这里可以看到，我们速度值是没有方向的，它永远是正值
        self.v_min = v_min
        self.v_max = v_max
        # 速度俯仰角范围：单位（°）
        self.theta_min = theta_min
        self.theta_max = theta_max
        # 速度方位角范围：单位（°）
        self.psi_min = psi_min
        self.psi_max = psi_max

        self.mode = mode
        # 如果mode为0，a,b,c分别表示nz, nx, Phi的范围，直接控制载荷，这里就没办法考虑加速度乃至速度超范围的情况
        # 如果mode为1，a,b,c分别表示velocity_a, theta_a, psi_a的范围。通过加速度来控制载荷，这里可以考虑载荷使得加速度和速度符合范围，但没办法考虑载荷是不是符合范围
        # 为了统一，这里所有的角度范围都要写成弧度
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.c_min = c_min
        self.c_max = c_max

        # ------------------- [修改后] -------------------
        # 目标状态初始化：改用球坐标系，保证距离在 100m - 5000m 之间均匀分布
        init_r = np.random.uniform(r_min, r_max)  # 距离：100m 到 5000m
        init_az = np.random.uniform(np.deg2rad(az_min), np.deg2rad(az_max))  # 方位角：-180° 到 180°
        init_el = np.random.uniform(np.deg2rad(el_min), np.deg2rad(el_max)) # 俯仰角：限制在合理空域 3° 到 60°

        # 转换为笛卡尔坐标系
        xc = init_r * np.cos(init_el) * np.cos(init_az)
        yc = init_r * np.cos(init_el) * np.sin(init_az)
        zc = init_r * np.sin(init_el)

        velocity = np.random.uniform(self.v_min, self.v_max)
        theta = np.random.uniform(self.theta_min, self.theta_max)
        psi = np.random.uniform(self.psi_min, self.psi_max)
        self.TC = self.target_control_init(flag=0, velocity=velocity, theta=theta)
        self.my_target = maneuvering_target_state_update(sT=sT, xc=xc, yc=yc, zc=zc, velocity=velocity,
                                                         theta=theta, psi=psi, nz=self.TC[0], nx=self.TC[1], Phi=self.TC[2],
                                                         anz=0.01, anx=0.01, aPhi=0.01,  sgz=0.01, sgx=0.01, sgp=0.1)

        self.SectionLen = SectionLen        #每一段的长度为SectionLen, 一段内载荷一致
        self.P_maneuver = P_maneuver        #每一段结束后，以概率P_maneuver做机动
        self.P_dead = P_dead                #每一段结束后，以概率P_dead消失
        self.MaxLastLen = MaxLastLen        #当轨迹点超过MaxLastLen，必然消失

        #全航迹列表
        self.trajectory = []
        self.judgment_list = [True, False]
        self.survival_p_list = [1-P_dead, P_dead]
        self.maneuver_p_list = [P_maneuver, 1-P_maneuver]

    # 目标状态初始化
    def target_control_init(self, flag=0, velocity=0, theta=0):
        if self.mode == 0:
            # 如果mode等于0, a是nz, b是nx, c是Phi
            nz = normal_distribution_truncation(self.a_min, self.a_max, 1.5)
            nx = normal_distribution_truncation(self.b_min, self.b_max, 1.5)
            Phi = normal_distribution_truncation(self.c_min, self.c_max, 1.5)
            velocity_a = 0
            theta_a = 0
            psi_a = 0
        elif self.mode == 1:
            # 如果mode等于1, a是velocity_a, b是theta_a, c是psi_a
            if flag == 1:
                velocity_a = normal_distribution_truncation(0, self.a_max, 1.5)
            elif flag == -1:
                velocity_a = normal_distribution_truncation(self.a_min, 0, 1.5)
            else:
                velocity_a = normal_distribution_truncation(self.a_min, self.a_max, 1.5)
            theta_a = normal_distribution_truncation(self.b_min, self.b_max, 1.5)
            psi_a = normal_distribution_truncation(self.c_min, self.c_max, 1.5)
            nz, nx, Phi = Acceleration2Load(velocity_a, theta_a, psi_a, velocity, theta)
        else:
            nz = 0
            nx = 0
            Phi = 0
            velocity_a = 0
            theta_a = 0
            psi_a = 0
        # 该目标控制状态设定如下：
        # [正常载荷, 切向载荷, 横滚角, 速度变化率, 速度俯仰角变化率, 速度方位角变化率]
        TC = np.array([0 for i in range(6)], 'float64')
        TC[0] = nz
        TC[1] = nx
        TC[2] = Phi
        TC[3] = velocity_a
        TC[4] = theta_a
        TC[5] = psi_a
        return TC

    # 单目标航迹片段生成
    def trajectory_section_generator(self):
        #[坐标x, 坐标y, 坐标z, 速度, 速度俯仰角, 速度方位角，正常载荷, 切向载荷, 横滚角, 速度变化率, 速度俯仰角变化率, 速度方位角变化率]
        trajectory_section = np.array([[0 for i in range(12)] for j in range(self.SectionLen)],'float64')
        for i in range(self.SectionLen):
            #生成新状态
            self.my_target.State_update_by_load_control()  # 更新了xc,yc,zc,velocity,theta,psi,velocity_a,theta_a,psi_a
            if self.my_target.zc <= 0:
                #目标着地，销毁目标
                return None
            if self.mode == 1:
                #判断速度范围：
                if not self.v_min < self.my_target.velocity < self.v_max:
                    self.TC[3] = 0              #速度超了，加速度应该为0
                #更新控制量——通过速度变化值
                self.TC[0], self.TC[1], self.TC[2] = Acceleration2Load(self.TC[3], self.TC[4], self.TC[5],
                                                                       self.my_target.velocity, self.my_target.theta)
            self.my_target.Load_control_update(self.TC[0], self.TC[1], self.TC[2])

            trajectory_section[i, 0] = self.my_target.xc
            trajectory_section[i, 1] = self.my_target.yc
            trajectory_section[i, 2] = self.my_target.zc
            trajectory_section[i, 3] = self.my_target.velocity
            trajectory_section[i, 4] = self.my_target.theta
            trajectory_section[i, 5] = self.my_target.psi
            trajectory_section[i, 6] = self.my_target.nz
            trajectory_section[i, 7] = self.my_target.nx
            trajectory_section[i, 8] = self.my_target.Phi
            trajectory_section[i, 9] = self.my_target.velocity_a
            trajectory_section[i, 10] = self.my_target.theta_a
            trajectory_section[i, 11] = self.my_target.psi_a
        return trajectory_section

    #全航迹生成
    def trajectory_generator(self):
        #航迹长度判断，是否超出最大长度，以及是否存活：
        target_survival = np.random.choice(a=self.judgment_list, p=self.survival_p_list)
        if len(self.trajectory) * self.SectionLen < self.MaxLastLen and target_survival:
            trajectory_section_new = self.trajectory_section_generator()
            #判断是否着地了，着地就不存活了。
            if trajectory_section_new is None:
                return False
            else:
                self.trajectory.append(trajectory_section_new)
                if np.random.choice(a=self.judgment_list, p=self.maneuver_p_list):
                    # print('velocity_a_0=%f' % (self.TC[3]))
                    # print('theta_a_0=%f' % (self.TC[4]))
                    # print('psi_a_0=%f' % (self.TC[5]))
                    # print('velocity_a_1=%f' % (self.my_target.velocity_a))
                    # print('theta_a_1=%f' % (self.my_target.theta_a))
                    # print('psi_a_1=%f' % (self.my_target.psi_a))
                    if self.my_target.velocity < self.v_min + 10:
                        flag = 1
                    elif self.my_target.velocity > self.v_max - 10:
                        flag = -1
                    else:
                        flag = 0
                    self.TC = self.target_control_init(flag=flag, velocity=self.my_target.velocity, theta=self.my_target.theta)
                    self.my_target.Load_control_update(self.TC[0], self.TC[1], self.TC[2])
                    # print('---------Maneuvering----------')
                return True
        else:
            return False

    #航迹输出——单目标
    def trajectory_output(self):
        n = 0
        while self.trajectory_generator():
            n += 1
        return n, np.concatenate(self.trajectory, axis=0)

class Trajectory_auto_generator_multitarget(object):
    def __init__(self, TargetNum=100):
        # 按目标均值计算，如果目标数量小于均值，则目标新生的数量要大于死亡数量，反之则新生目标数量小于死亡数量。
        self.TargetNum = TargetNum          #目标的数量的均值

        #目标航迹列表
        self.targets_list = []
        for i in range(self.TargetNum):
            target_t = Trajectory_auto_generator_one_target()
            self.targets_list.append([target_t, 0]) #0代表起始step
        #死亡目标维护
        self.dead_list = []

        # ===== 新增这一行 =====
        # 保存默认的 SectionLen，防止 targets_list 为空时出错
        self.default_section_len = self.targets_list[0][0].SectionLen if self.targets_list else 20
    def Multitarget_trajectories_one_step(self):
        targets_list_new = []
        for tg in self.targets_list:
            tg_survival = tg[0].trajectory_generator()
            if tg_survival:
                targets_list_new.append(tg)
            else:
                self.dead_list.append(tg)
        self.targets_list.clear()
        self.targets_list = targets_list_new.copy()
        targets_list_new.clear()

    #生成新目标
    def Target_born(self, deta_num, step):
        new_target_num = np.random.randint(-deta_num, deta_num*2)
        if new_target_num > 0:
            for i in range(new_target_num):
                target_t = Trajectory_auto_generator_one_target()
                self.targets_list.append([target_t, step])

    #连续生成多目标航迹
    def Multitarget_trajectories(self, steps=10):
        for i in range(steps):
            #每个目标生成一段航迹
            self.Multitarget_trajectories_one_step()
            deta_num = self.TargetNum - len(self.targets_list)
            if deta_num > 0:
                # print(deta_num)
                self.Target_born(deta_num=deta_num, step=i+1)

    #截断观测——输入观测的起始点和结束点，输出这段时间的观测以及对应的航迹
    def Multitarget_observations(self, begin_step, end_step, sg_theta, sg_dis):
        #多目标运动模拟，观测值的单位是°
        self.Multitarget_trajectories(steps=end_step)
        #找到所有这个时间节点内的航迹，为观测做准备
        Trajectories = []
        #有死亡的目标，判断这些目标的航迹有没有在规定范围内
        if len(self.dead_list) > 0:
            for dt in self.dead_list:
                trajectory_t = dt[0].trajectory
                if len(trajectory_t) == 0:     #空航迹，返回
                    continue
                if (begin_step < dt[1] + len(trajectory_t) and dt[1] < end_step):          #绝对时间
                    bs_dt = begin_step-dt[1]
                    trajectory_truncation = trajectory_t[max(bs_dt, 0):min(end_step-dt[1],len(trajectory_t))]
                    if bs_dt < 0:
                        # begin_time = - bs_dt * self.targets_list[0][0].SectionLen
                        begin_time = - bs_dt * self.default_section_len
                    else:
                        begin_time = 0
                    Trajectories.append([np.concatenate(trajectory_truncation, axis=0),begin_time])  #相对时间
        #对于存活的目标，判断这些目标的航迹有没有在规定范围内
        if len(self.targets_list) > 0:
            for st in self.targets_list:
                trajectory_t = st[0].trajectory
                if (begin_step < st[1] + len(trajectory_t) and st[1] < end_step):          #绝对时间
                    bs_dt = begin_step-st[1]
                    trajectory_truncation = trajectory_t[max(bs_dt, 0):min(end_step-st[1],len(trajectory_t))]
                    if bs_dt < 0:
                        # begin_time = - bs_dt * self.targets_list[0][0].SectionLen
                        begin_time = - bs_dt * self.default_section_len
                    else:
                        begin_time = 0
                    Trajectories.append([np.concatenate(trajectory_truncation, axis=0),begin_time])  #相对时间
        #每一条航迹生成观测值
        Observations_of_each_trajectory = []
        for this_trajectory in Trajectories:
            Tj = this_trajectory[0]
            #航迹去掉初始值
            TN = np.size(Tj, axis=0)
            #观测值
            Obser = np.array([[0 for i in range(3)] for j in range(TN)],'float64') #Initialization of observation
            #Observations without noise
            Obser[:, 2] = np.sqrt(np.square(Tj[:, 0]) + np.square(Tj[:, 1]) + np.square(Tj[:, 2]))  # Distance
            Obser[:, 0] = np.arcsin(Tj[:, 2]/Obser[:, 2])*180/np.pi      #Pitch angle 单位°
            Obser[:, 1] = np.arctan2(Tj[:, 1], Tj[:, 0])*180/np.pi  #Azimuth angle 单位°
            if sg_theta is None:
                Observations_of_each_trajectory.append([Obser, this_trajectory[1]])
            else:
                # Observations with noise
                Obser_n = np.array([[0 for i in range(3)] for j in range(TN)], 'float64')
                Obser_n[:, 0] = Obser[:, 0] + np.random.normal(0, sg_theta, TN)  # Pitch angle
                Obser_n[:, 1] = Obser[:, 1] + np.random.normal(0, sg_theta, TN)  # Azimuth angle
                Obser_n[:, 2] = Obser[:, 2] + np.random.normal(0, sg_dis, TN)  # Distance
                Observations_of_each_trajectory.append([Obser, Obser_n, this_trajectory[1]])
        return Trajectories, Observations_of_each_trajectory

    #多传感器观测
    def Multitarget_multiobservations(self, TimeStepRange=[0,10], Camera=None, Radar=None, RFsensor=None):
        # TimeStepRange=[begin_step, end_step]
        # Camera=[MaxR, sg_theta]  MaxR摄像头检测的最大距离，与1个像素点的最远分辨距离对应。 sg_theta为角度检测偏差，与焦距和分辨率相关，设为0.002度合适
        # 摄像头监控角度小，发现目标的概率低，发现目标的概率与距离的平方成反比
        # Radar=[MinR, MaxR, sg_theta, sg_dis, PD]  MinR雷达最近检测距离，MaxR雷达最远检测距离。sg_theta为角度检测偏差，一般为0.5度，sg_dis是距离检测偏差，一般为10m
        # 雷达全空域扫描的速度比较快，目标的发现概率PD比较高,设定为0.9，但是存在杂波
        # RFsensor=[MaxR, sg_theta]  MaxR射频检测的最大距离。 sg_theta为角度检测偏差，这个方差较大，可以5度
        # 被动射频检测，默认无漏警与预警率
        # 如果是None，对应的量测为空

        #生成航迹以及对应的无噪观测(先不考虑高度大于0的问题)
        Trajectories, Observations = self.Multitarget_observations(begin_step=TimeStepRange[0], end_step=TimeStepRange[1], sg_theta=None, sg_dis=None)
        Camera_observations = []
        Radar_observations = []
        RFsensor_observations = []
        judgment_list = [True, False]
        for obser_n in Observations:
            camera_obser = []
            radar_obser = []
            rfsensor_obser = []
            obser_L = np.size(obser_n[0],axis=0)
            for i in range(obser_L):
                target_distance = obser_n[0][i,2]

                # 相机观测结果：在观测范围内
                if (Camera is not None) and (target_distance < Camera[0]):
                    #假设的相机观测机制-----------------------------------------------------------
                    target_length = 0.5  # 假设目标长度（m）
                    spherical_surface = 2 * np.pi * (target_distance ** 2)  # 目标所在球面面积(半个球面)
                    camera_scan_P = 1 / 16  # 为了简单起见，我们这里直接假设没一次检测，相机能扫描到camera_scan_P比例的球面
                    spherical_surface_contain_target = (np.sqrt(
                        spherical_surface * camera_scan_P) + target_length) ** 2  # 一次检测能看到目标的面积（做最简单的假设了）
                    camera_p = min(1, spherical_surface_contain_target / spherical_surface)  # 相机发现目标的概率，简单假设一下
                    camera_find_target = [camera_p, 1-camera_p]
                    #--------------------------------------------------------------------------
                    if np.random.choice(a=judgment_list, p=camera_find_target):
                        camera_data = np.zeros(2)
                        camera_data[0] = obser_n[0][i, 0] + np.random.normal(0, Camera[1])
                        camera_data[1] = obser_n[0][i, 1] + np.random.normal(0, Camera[1])
                    else:
                        camera_data = []
                else:
                    camera_data = []

                #雷达观测结果：在观测范围内
                if (Radar is not None) and (Radar[0] < target_distance < Radar[1]):
                    radar_find_target = [Radar[4], 1-Radar[4]]
                    #--------------------------------------------------------------------------
                    if np.random.choice(a=judgment_list, p=radar_find_target):
                        radar_data = np.zeros(3)
                        radar_data[0] = obser_n[0][i, 0] + np.random.normal(0, Radar[2])
                        radar_data[1] = obser_n[0][i, 1] + np.random.normal(0, Radar[2])
                        radar_data[2] = obser_n[0][i, 2] + np.random.normal(0, Radar[3])
                    else:
                        radar_data = []
                else:
                    radar_data = []

                #射频观测结果：在观测范围内
                # 增加了漏警率判断 Logic
                if (RFsensor is not None) and (target_distance < RFsensor[0]):
                    # 获取检测概率 PD，如果用户没传第三个参数，默认给 0.95
                    rf_pd = RFsensor[2] if len(RFsensor) > 2 else 0.95
                    rf_find_target_prob = [rf_pd, 1 - rf_pd]

                    # 使用 random.choice 进行概率判断（检测 or 漏警）
                    if np.random.choice(a=judgment_list, p=rf_find_target_prob):
                        rf_data = np.zeros(2)
                        rf_data[0] = obser_n[0][i, 0] + np.random.normal(0, RFsensor[1])  # Pitch
                        rf_data[1] = obser_n[0][i, 1] + np.random.normal(0, RFsensor[1])  # Azimuth
                    else:
                        rf_data = []  # 漏警：虽然在距离内，但没检测到
                else:
                    rf_data = []  # 超出距离

                camera_obser.append(camera_data)
                radar_obser.append(radar_data)
                rfsensor_obser.append(rf_data)

            Camera_observations.append(camera_obser)
            Radar_observations.append(radar_obser)
            RFsensor_observations.append(rfsensor_obser)
        return Trajectories, Observations, Camera_observations, Radar_observations, RFsensor_observations

    def radar_clutter_generator(self, radar_detection_range, clutter_num_mean):
        """
        [修改] 真实雷达杂波模拟器
        1. 距离: 采用截断指数分布，模拟近处杂波密集、远处杂波稀疏的特性。
        2. 俯仰角: 采用 Beta 分布，模拟低仰角（地杂波）密集特性。
        """
        clutter_num = max(int(np.random.normal(clutter_num_mean, 5)), 0)
        clutters = []
        for i in range(clutter_num):
            clutter_data = np.zeros(3)

            # 1. 俯仰角 (Pitch): 使用 Beta(1.5, 8) 分布，极度偏向于 0-15 度 (地杂波)
            # 映射到 0.5 到 80 度
            clutter_data[0] = np.random.beta(1.5, 8) * 80 + 0.5

            # 2. 方位角 (Azimuth): 全向雷达，杂波方位均匀分布
            clutter_data[1] = np.random.uniform(-180, 180)

            # 3. 距离 (Range): 指数衰减分布
            # scale 控制衰减速度，设为最大距离的 1/4，使大部分杂波落在近场
            scale = (radar_detection_range[1] - radar_detection_range[0]) / 4.0
            r = np.random.exponential(scale) + radar_detection_range[0]
            # 截断处理：确保不超过最大检测距离
            while r > radar_detection_range[1]:
                r = np.random.exponential(scale) + radar_detection_range[0]

            clutter_data[2] = r
            clutters.append(clutter_data)
        return clutters

    def rf_clutter_generator(self, active_rf_targets, multipath_prob=0.3, false_alarm_rate=0.2):
        """
        生成RF杂波：包括基于真实目标的多径鬼影(Multipath Ghosts)和随机干扰(False Alarms)
        active_rf_targets: list of [Pitch, Azimuth]，当前时刻真实检测到的RF目标
        """
        clutters = []

        # 1. 多径效应 (Multipath Ghosts) —— 依赖于真实目标
        for target in active_rf_targets:
            # target[0] 是 Pitch, target[1] 是 Azimuth
            if np.random.rand() < multipath_prob:
                # 鬼影通常在方位角上有较大偏移 (例如 20-60度)
                offset = np.random.uniform(20, 60) * np.random.choice([1, -1])
                ghost_az = target[1] + offset

                # 归一化方位角到 -180 ~ 180
                if ghost_az > 180:
                    ghost_az -= 360
                elif ghost_az < -180:
                    ghost_az += 360

                # 鬼影的俯仰角通常会由于反射变得不稳定
                ghost_el = target[0] + np.random.normal(0, 5)
                ghost_el = np.clip(ghost_el, 0, 90)  # 限制在合理范围

                clutter_data = np.zeros(2)
                clutter_data[0] = ghost_el
                clutter_data[1] = ghost_az
                clutters.append(clutter_data)
        # 2. 随机环境干扰 (Interference) —— 独立随机虚警
        # 模拟 Wi-Fi 或其他同频干扰
        if np.random.rand() < false_alarm_rate:
            num_interference = np.random.randint(1, 4)  # 生成1-3个干扰点
            for _ in range(num_interference):
                noise_data = np.zeros(2)
                noise_data[0] = np.random.uniform(0, 30)  # 干扰通常来自地面或低空
                noise_data[1] = np.random.uniform(-180, 180)  # 方位随机
                clutters.append(noise_data)

        return clutters

    def show_data(self, TimeStepRange, Camera=None, Radar=None, RFsensor=None, clutter_num_mean=1):
        # 数据生成
        Trajectories, Observations, Camera_observations, Radar_observations, RFsensor_observations = self.Multitarget_multiobservations(
            TimeStepRange=TimeStepRange, Camera=Camera, Radar=Radar, RFsensor=RFsensor)

        # 1. 生成雷达杂波 (Clutters_in_radar)
        Clutters_in_radar = []
        # time_step_num = (TimeStepRange[1] - TimeStepRange[0]) * self.targets_list[0][0].SectionLen
        time_step_num = (TimeStepRange[1] - TimeStepRange[0]) * self.default_section_len
        if Radar is not None:
            for i in range(time_step_num):
                Clutters_in_radar.append(self.radar_clutter_generator(radar_detection_range=[Radar[0], Radar[1]],
                                                                      clutter_num_mean=clutter_num_mean))
        else:
            Clutters_in_radar = None

        # 2. 生成射频杂波 (Clutters_in_rfsensor)
        Clutters_in_rfsensor = []
        if RFsensor is not None:
            for i in range(time_step_num):
                # 第一步：需要找出当前时间步(i)有哪些“真实”的RF目标被检测到了
                # 因为只有存在真实强信号时，多径效应(鬼影)才会显著
                current_active_rf_targets = []

                # 遍历所有轨迹，找到对应时间步 i 的RF观测值
                for traj_idx in range(len(Observations)):
                    # 获取该航迹的开始时间(相对时间)
                    start_time = int(Observations[traj_idx][1])
                    # 计算当前绝对时间 i 在该航迹中的索引
                    curr_traj_idx = i - start_time

                    # 检查索引是否有效，且该航迹在此时刻是否有RF数据
                    if 0 <= curr_traj_idx < len(RFsensor_observations[traj_idx]):
                        rf_meas = RFsensor_observations[traj_idx][curr_traj_idx]
                        if len(rf_meas) > 0:  # 类似于 [] 代表没检测到
                            current_active_rf_targets.append(rf_meas)

                # 第二步：调用RF杂波生成器
                # 参数可调整：multipath_prob(鬼影概率), false_alarm_rate(随机干扰概率)
                rf_clutters = self.rf_clutter_generator(current_active_rf_targets,
                                                        multipath_prob=0.3,
                                                        false_alarm_rate=0.2)
                Clutters_in_rfsensor.append(rf_clutters)
        else:
            Clutters_in_rfsensor = None
        return Trajectories, Observations, Camera_observations, Radar_observations, RFsensor_observations, Clutters_in_radar, Clutters_in_rfsensor

def robust_list_to_object_array(nested_list):
    """
    更健壮的列表到对象数组转换函数，处理不规则结构
    """
    if not nested_list:
        return np.array([], dtype=object)

    # 处理一维列表
    if not isinstance(nested_list[0], (list, np.ndarray)):
        return np.array(nested_list, dtype=object)

    # 处理二维列表
    rows = len(nested_list)

    # 找出最大列数，处理不规则的二维结构
    max_cols = max(len(row) for row in nested_list) if rows > 0 else 0

    # 创建对象数组
    obj_array = np.empty((rows, max_cols), dtype=object)

    # 填充数据，处理不规则的列数
    for i in range(rows):
        for j in range(len(nested_list[i])):
            element = nested_list[i][j]
            if not isinstance(element, np.ndarray):
                obj_array[i, j] = np.array(element)
            else:
                obj_array[i, j] = element
        # 对于列数不足的行，剩余位置填充None
        for j in range(len(nested_list[i]), max_cols):
            obj_array[i, j] = np.array([])

    return obj_array


def Observation_Lables(Observations, Radar_observations, RFsensor_observations, Clutters_in_radar,
                       Clutters_in_rfsensor):
    """
    分别为雷达和射频传感器生成融合数据（观测+杂波）和对应的标签。
    将真实观测数据按照时间戳随机插入到对应时刻的杂波列表中。

    参数:
        Observations: 观测数据列表，用于获取航迹起始时间
        Radar_observations: 雷达真实观测
        RFsensor_observations: 射频真实观测
        Clutters_in_radar: 雷达杂波
        Clutters_in_rfsensor: 射频杂波

    返回:
        Fuse_Radar_Data, Fuse_Radar_Labels, Fuse_RF_Data, Fuse_RF_Labels
    """

    # 1. 初始化雷达融合数据和标签
    # 标签约定：0 代表杂波/虚警，i+1 代表第 i 条真实航迹
    time_steps = len(Clutters_in_radar)
    Fuse_Radar_Data = copy.deepcopy(Clutters_in_radar)
    # 初始化标签列表，对应现有的杂波，标签全为 0
    Fuse_Radar_Labels = [[0] * len(sublist) for sublist in Fuse_Radar_Data]

    # 2. 初始化射频融合数据和标签
    Fuse_RF_Data = copy.deepcopy(Clutters_in_rfsensor) if Clutters_in_rfsensor is not None else []
    if Clutters_in_rfsensor is not None:
        Fuse_RF_Labels = [[0] * len(sublist) for sublist in Fuse_RF_Data]
    else:
        Fuse_RF_Labels = []

    # 3. 遍历每一条真实航迹，将其观测值插入到融合列表中
    for i in range(len(Observations)):
        # 获取航迹开始时刻 (Observations[i][1] 是时间步索引)
        start_time = int(Observations[i][1])
        target_label_id = i + 1  # 航迹ID从1开始，便于区分杂波0

        # --- 处理雷达融合 ---
        if i < len(Radar_observations):
            radar_obs_seq = Radar_observations[i]
            for j in range(len(radar_obs_seq)):
                time_step = start_time + j

                # 越界检查 & 有效性检查
                if time_step < time_steps and len(radar_obs_seq[j]) > 0:
                    # 获取该时刻的数据列表
                    current_step_data = Fuse_Radar_Data[time_step]
                    current_step_labels = Fuse_Radar_Labels[time_step]

                    # 随机位置插入
                    if len(current_step_data) > 0:
                        position = random.randint(0, len(current_step_data))
                    else:
                        position = 0

                    current_step_data.insert(position, radar_obs_seq[j])
                    current_step_labels.insert(position, target_label_id)

        # --- 处理射频融合 ---
        if Clutters_in_rfsensor is not None and i < len(RFsensor_observations):
            rf_obs_seq = RFsensor_observations[i]
            for j in range(len(rf_obs_seq)):
                time_step = start_time + j

                # 越界检查 & 有效性检查
                if time_step < len(Fuse_RF_Data) and len(rf_obs_seq[j]) > 0:
                    # 获取该时刻的数据列表
                    current_step_data = Fuse_RF_Data[time_step]
                    current_step_labels = Fuse_RF_Labels[time_step]

                    # 随机位置插入
                    if len(current_step_data) > 0:
                        position = random.randint(0, len(current_step_data))
                    else:
                        position = 0

                    current_step_data.insert(position, rf_obs_seq[j])
                    current_step_labels.insert(position, target_label_id)

    return Fuse_Radar_Data, Fuse_Radar_Labels, Fuse_RF_Data, Fuse_RF_Labels


if __name__ == '__main__':
    TN = 3  # 平均目标数
    TSR = [10, 110]  # 检测航迹片段的时间范围，这个数值是片段数
    C_ = [10000, 0.002]  # 相机参数[最大检测距离（m），角度检测偏差（°）]
    R_ = [100, 5000, 0.25, 3.5, 0.8]  # 雷达参数
    RF = [10000, 2.5, 0.9]  # 射频参数[最大检测距离, 角度偏差, 检测概率]

    multitargets = Trajectory_auto_generator_multitarget(TargetNum=TN)

    # 1. 生成原始数据和杂波
    Trajectories, Observations, Camera_observations, Radar_observations, RFsensor_observations, Clutters_in_radar, Clutters_in_rfsensor = multitargets.show_data(
        TimeStepRange=TSR, Camera=C_, Radar=R_, RFsensor=RF, clutter_num_mean=5)

    # 2. 生成融合数据（插入标签）
    # --- 修改点：传入 Clutters_in_rfsensor，并接收 4 个返回值 ---
    Fuse_Radar_Data, Fuse_Radar_Labels, Fuse_RF_Data, Fuse_RF_Labels = Observation_Lables(
        Observations, Radar_observations, RFsensor_observations, Clutters_in_radar, Clutters_in_rfsensor)

    # 3. 格式转换 (List -> Object Array)
    Trajectories = robust_list_to_object_array(Trajectories)
    Observations = robust_list_to_object_array(Observations)
    Camera_observations = robust_list_to_object_array(Camera_observations)
    Radar_observations = robust_list_to_object_array(Radar_observations)
    RFsensor_observations = robust_list_to_object_array(RFsensor_observations)
    Clutters_in_radar = robust_list_to_object_array(Clutters_in_radar)
    Clutters_in_rfsensor = robust_list_to_object_array(Clutters_in_rfsensor)

    # --- 修改点：新增融合数据的转换 ---
    Fuse_Radar_Data = robust_list_to_object_array(Fuse_Radar_Data)
    Fuse_Radar_Labels = robust_list_to_object_array(Fuse_Radar_Labels)
    Fuse_RF_Data = robust_list_to_object_array(Fuse_RF_Data)
    Fuse_RF_Labels = robust_list_to_object_array(Fuse_RF_Labels)

    # 4. 保存数据
    mydata = {
        'Trajectories': Trajectories,
        'Observations': Observations,
        'Camera_observations': Camera_observations,
        'Radar_observations': Radar_observations,
        'RFsensor_observations': RFsensor_observations,
        'Clutters_in_radar': Clutters_in_radar,
        'Clutters_in_rfsensor': Clutters_in_rfsensor,
        # --- 保存融合后的数据 ---
        'Fuse_Radar_Data': Fuse_Radar_Data,
        'Fuse_Radar_Labels': Fuse_Radar_Labels,
        'Fuse_RF_Data': Fuse_RF_Data,
        'Fuse_RF_Labels': Fuse_RF_Labels
    }
    scio.savemat('trajectory_samples.mat', mydata)

