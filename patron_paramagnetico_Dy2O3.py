#%%
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet 
import re
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy 
from scipy.optimize import curve_fit 
from scipy.stats import linregress
from uncertainties import ufloat, unumpy
from glob import glob

#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                else:
                    # Capturar los casos con nombres de archivo en las últimas dos líneas
                    match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                    if match_files:
                        key = match_files.group(1)[2:]  # Obtener el nombre de la clave sin '# '
                        value = match_files.group(2)     # Obtener el nombre del archivo
                        meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=18,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N


#%%
# pendientes_212 = glob(os.path.join('./212_150', '**', 'pendientes.txt'),recursive=True)
# pendientes_238 = glob(os.path.join('./238_150', '**', 'pendientes.txt'),recursive=True)
# pendientes_265 = glob(os.path.join('./265_150', '**', 'pendientes.txt'),recursive=True)
# pendientes_212 = glob(os.path.join('./212_150', '**', 'pendientes.txt'),recursive=True)

pendientes_100 = glob(os.path.join('./100kHz', '**', 'pendientes.txt'),recursive=True)
pendientes_300 = glob(os.path.join('./300kHz', '**', 'pendientes.txt'),recursive=True)

resultados_100 = glob(os.path.join('./100kHz', '**', '*resultados.txt'),recursive=True)
resultados_300 = glob(os.path.join('./300kHz', '**', '*resultados.txt'),recursive=True)
#%%
# all_212=[]
# for f in pendientes_212:
#     all_212.append(np.loadtxt(f,dtype='float'))
# conc_212=np.concatenate(all_212,axis=0)    
# mean_212=np.mean(conc_212)
# std_212=np.std(conc_212)
# Pendiente_212=ufloat(mean_212,std_212)
# print('-'*40,'\n',f'Pendiente 212kHz = {Pendiente_212:.2e} A/mVs ({len(conc_212)} muestras)')

# all_238=[]
# for f in pendientes_238:
#     all_238.append(np.loadtxt(f,dtype='float'))
# conc_238=np.concatenate(all_238,axis=0)    
# mean_238=np.mean(conc_238)
# std_238=np.std(conc_238)
# Pendiente_238=ufloat(mean_238,std_238)
# print('-'*40,'\n',f'Pendiente 238kHz = {Pendiente_238:.2e} A/mVs ({len(conc_238)} muestras)')

# all_265=[]
# for f in pendientes_265:
#     all_265.append(np.loadtxt(f,dtype='float'))
# conc_265=np.concatenate(all_265,axis=0)    
# mean_265=np.mean(conc_265)
# std_265=np.std(conc_265)
# Pendiente_265=ufloat(mean_265,std_265)
# print('-'*40,'\n',f'Pendiente 265kHz = {Pendiente_265:.2e} A/mVs ({len(conc_265)} muestras)')


all_100=[]
for f in pendientes_100:
    all_100.append(np.loadtxt(f,dtype='float'))
conc_100=np.concatenate(all_100,axis=0)    
mean_100=np.mean(conc_100)
std_100=np.std(conc_100)
Pendiente_100=ufloat(mean_100,std_100)
print('-'*40,'\n',f'Pendiente 100kHz = {Pendiente_100:.2e} A/mVs ({len(conc_100)} muestras)')


all_300=[]
for f in pendientes_300:
    all_300.append(np.loadtxt(f,dtype='float'))
conc_300=np.concatenate(all_300,axis=0)    
mean_300=np.mean(conc_300)
std_300=np.std(conc_300)
Pendiente_300=ufloat(mean_300,std_300)
print('-'*40,'\n',f'Pendiente 300kHz = {Pendiente_300:.2e} A/mVs ({len(conc_300)} muestras)')

# pendientes_all = np.concatenate([conc_212,conc_238,conc_265,conc_300])
# mean_all=np.mean(pendientes_all)
# std_all=np.std(pendientes_all)
# pend_all=ufloat(mean_all,std_all)
# print('-'*40,'\n',f'Pendiente promedio para las 4 frecuencias = {pend_all:.2e} A/mVs ({len(pendientes_all)} muestras)')

#%%

all_res_100=[]
for f in resultados_100:
    _, _, _,_,  _, _, _, mag_max, _, _, _ , _, _, _, _=lector_resultados(f)
    all_res_100.append(mag_max)
conc_100=np.concatenate(all_res_100,axis=0)    
mean_100=np.mean(conc_100)
std_100=np.std(conc_100)
mag_max_100=ufloat(mean_100,std_100)
print('-'*40,'\n',f'mag_max 100kHz = {mag_max_100:.1f} A/m ({len(conc_100)} muestras)')



all_res_300=[]
for f in resultados_300:
    _, _, _,_,  _, _, _, mag_max, _, _, _ , _, _, _, _=lector_resultados(f)
    all_res_300.append(mag_max)
conc_300=np.concatenate(all_res_300,axis=0)    
mean_300=np.mean(conc_300)
std_300=np.std(conc_300)
mag_max_300=ufloat(mean_300,std_300)
print('-'*40,'\n',f'mag_max 300kHz = {mag_max_300:.1f} A/m ({len(conc_300)} muestras)')
#%%
all_res_300=[]
for f in resultados_300:
    all_res_300.append(np.loadtxt(f,dtype='float'))
conc_300=np.concatenate(all_res_300,axis=0)    
mean_300=np.mean(conc_300)
std_300=np.std(conc_300)
Pendiente_300=ufloat(mean_300,std_300)
print('-'*40,'\n',f'Pendiente 300kHz = {Pendiente_300:.2e} A/mVs ({len(conc_300)} muestras)')


# %%
# t_135_1,H_Vs_135_1,M_Vs_135_1,H_kAm_135_1,M_Am_135_1,_= lector_ciclos(ciclos_135[0])
# t_135_2,H_Vs_135_2,M_Vs_135_2,H_kAm_135_2,M_Am_135_2,_= lector_ciclos(ciclos_135[1])
# t_135_3,H_Vs_135_3,M_Vs_135_3,H_kAm_135_3,M_Am_135_3,_= lector_ciclos(ciclos_135[2])

# t_212_1,H_Vs_212_1,M_Vs_212_1,H_kAm_212_1,M_Am_212_1,_= lector_ciclos(ciclos_212[0])
# t_212_2,H_Vs_212_2,M_Vs_212_2,H_kAm_212_2,M_Am_212_2,_= lector_ciclos(ciclos_212[1])
# t_212_3,H_Vs_212_3,M_Vs_212_3,H_kAm_212_3,M_Am_212_3,_= lector_ciclos(ciclos_135[2])

# t_300_1,H_Vs_300_1,M_Vs_300_1,H_kAm_300_1,M_Am_300_1,_= lector_ciclos(ciclos_300[0])
# t_300_2,H_Vs_300_2,M_Vs_300_2,H_kAm_300_2,M_Am_300_2,_= lector_ciclos(ciclos_300[1])
# t_300_3,H_Vs_300_3,M_Vs_300_3,H_kAm_300_3,M_Am_300_3,_= lector_ciclos(ciclos_300[2])
# #%%
# fig,ax = plt.subplots(figsize=(8,6),constrained_layout=True)
# ax.plot(H_kAm_135_1,M_Vs_135_1)
# ax.plot(H_kAm_135_2,M_Vs_135_2)
# ax.plot(H_kAm_135_3,M_Vs_135_3)

# ax.plot(H_kAm_212_1,M_Vs_212_1)
# ax.plot(H_kAm_212_2,M_Vs_212_2)
# ax.plot(H_kAm_212_3,M_Vs_212_3)

# ax.plot(H_kAm_300_1,M_Vs_300_1)
# ax.plot(H_kAm_300_2,M_Vs_300_2)
# ax.plot(H_kAm_300_3,M_Vs_300_3)

# ax.grid()
# ax.set_ylabel('M (V*s)')
# ax.set_xlabel('H (A/m)')
# #%% Ajuste lineal sobre cada ciclo
# def lineal(x,m,n):
#     return m*x+n

# (m_135_1,n_135_1),_ = curve_fit(f=lineal, xdata=H_kAm_135_1,ydata=M_Vs_135_1)
# (m_135_2,n_135_2),_ = curve_fit(f=lineal, xdata=H_kAm_135_2,ydata=M_Vs_135_2)
# (m_135_3,n_135_3),_ = curve_fit(f=lineal, xdata=H_kAm_135_3,ydata=M_Vs_135_3)

# (m_212_1,n_212_1),_ = curve_fit(f=lineal, xdata=H_kAm_212_1,ydata=M_Vs_212_1)
# (m_212_2,n_212_2),_ = curve_fit(f=lineal, xdata=H_kAm_212_2,ydata=M_Vs_212_2)
# (m_212_3,n_212_3),_ = curve_fit(f=lineal, xdata=H_kAm_212_3,ydata=M_Vs_212_3)

# (m_300_1,n_300_1),_ = curve_fit(f=lineal, xdata=H_kAm_300_1,ydata=M_Vs_300_1)
# (m_300_2,n_300_2),_ = curve_fit(f=lineal, xdata=H_kAm_300_2,ydata=M_Vs_300_2)
# (m_300_3,n_300_3),_ = curve_fit(f=lineal, xdata=H_kAm_300_3,ydata=M_Vs_300_3)

# m_mean  = np.mean(np.array([m_135_1,m_135_2,m_135_3,m_212_1,m_212_2,m_212_3,m_300_1,m_300_2,m_300_3]))
# m_std = np.std(np.array([m_135_1,m_135_2,m_135_3,m_212_1,m_212_2,m_212_3,m_300_1,m_300_2,m_300_3]))
# m =ufloat(m_mean,m_std)
# print(f'Pendiente media = {m:.2e} Vs/A/m')
# n_mean  = np.mean(np.array([n_135_1,n_135_2,n_135_3,n_212_1,n_212_2,n_212_3,n_300_1,n_300_2,n_300_3]))

# #%%
# x_new= np.linspace(-57712,57712,100)
# y_new= lineal(x_new,m_mean,n_mean)

# fig,ax = plt.subplots(constrained_layout=True)
# ax.plot(H_kAm_135_1,M_Vs_135_1)
# ax.plot(H_kAm_135_2,M_Vs_135_2)
# ax.plot(H_kAm_135_3,M_Vs_135_3)

# ax.plot(H_kAm_212_1,M_Vs_212_1)
# ax.plot(H_kAm_212_2,M_Vs_212_2)
# ax.plot(H_kAm_212_3,M_Vs_212_3)

# ax.plot(H_kAm_300_1,M_Vs_300_1)
# ax.plot(H_kAm_300_2,M_Vs_300_2)
# ax.plot(H_kAm_300_3,M_Vs_300_3)

# ax.plot(x_new,y_new,'o-',label=f'$< m > =${m} Vs/A/m')
# ax.legend()

# ax.grid()
# ax.set_ylabel('M (V*s)')
# ax.set_xlabel('H (A/m)')
# ax.set_title('Patron Dy$_2$O$_3$')
# plt.show()