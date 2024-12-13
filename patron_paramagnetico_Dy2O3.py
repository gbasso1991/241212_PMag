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


#%% Pendientes
pendientes_100 = glob(os.path.join('./100kHz', '**', 'pendientes.txt'),recursive=True)
pendientes_300 = glob(os.path.join('./300kHz', '**', 'pendientes.txt'),recursive=True)

resultados_100 = glob(os.path.join('./100kHz', '**', '*resultados.txt'),recursive=True)
resultados_300 = glob(os.path.join('./300kHz', '**', '*resultados.txt'),recursive=True)

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

#%% Mag Max
all_res_100=[]
for f in resultados_100:
    _, _, _,_,  _, _, _, mag_max, _, _, _ , _, _, _, _=lector_resultados(f)
    all_res_100.append(mag_max)
conc_res_100=np.concatenate(all_res_100,axis=0)    
mean_100=np.mean(conc_res_100)
std_100=np.std(conc_res_100)
mag_max_100=ufloat(mean_100,std_100)
print('-'*40,'\n',f'mag_max 100kHz = {mag_max_100:.1f} A/m ({len(conc_res_100)} muestras)')
all_res_300=[]
for f in resultados_300:
    _, _, _,_,  _, _, _, mag_max, _, _, _ , _, _, _, _=lector_resultados(f)
    all_res_300.append(mag_max)
conc_res_300=np.concatenate(all_res_300,axis=0)    
mean_300=np.mean(conc_res_300)
std_300=np.std(conc_res_300)
mag_max_300=ufloat(mean_300,std_300)
print('-'*40,'\n',f'mag_max 300kHz = {mag_max_300:.1f} A/m ({len(conc_res_300)} muestras)')


# %% Ploteo ciclos
ciclos_100 = glob(os.path.join('100kHz','**', '*ciclo_promedio*'),recursive=True)
ciclos_100.sort()
labels_100 = ['1','2','3','4']

ciclos_300 = glob(os.path.join('300kHz','**', '*ciclo_promedio*'),recursive=True)
ciclos_300.sort()
labels_300 = ['1','2','3','4']

#%%
_,_,_,H100_1,M_100_1,meta_100_1=lector_ciclos(ciclos_100[0]) 
_,_,_,H100_2,M_100_2,meta_100_2=lector_ciclos(ciclos_100[1]) 
_,_,_,H100_3,M_100_3,meta_100_3=lector_ciclos(ciclos_100[2]) 
_,_,_,H100_4,M_100_4,meta_100_4=lector_ciclos(ciclos_100[3])


_,_,_,H300_1,M_300_1,meta_300_1=lector_ciclos(ciclos_300[0]) 
_,_,_,H300_2,M_300_2,meta_300_2=lector_ciclos(ciclos_300[1]) 
_,_,_,H300_3,M_300_3,meta_300_3=lector_ciclos(ciclos_300[2]) 
_,_,_,H300_4,M_300_4,meta_300_4=lector_ciclos(ciclos_300[3])


#%%

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5), constrained_layout=True,sharey=True)

# Ciclos 100
ax1.plot(H100_1, M_100_1, label=labels_100[0])
ax1.plot(H100_2, M_100_2, label=labels_100[1])
ax1.plot(H100_3, M_100_3, label=labels_100[2])
ax1.plot(H100_4, M_100_4, label=labels_100[3])
ax1.text(0.75,1/3,f'M max = {mag_max_100:.0f} A/m\n{len(conc_res_100)} muestras',bbox=dict(facecolor='tab:blue', alpha=0.5),transform=ax1.transAxes,va='center',ha='center')

ax1.set_title('100 kHz', fontsize=14)


# Ciclos 300
ax2.plot(H300_1, M_300_1, label=labels_300[0])
ax2.plot(H300_2, M_300_2, label=labels_300[1])
ax2.plot(H300_3, M_300_3, label=labels_300[2])
ax2.plot(H300_4, M_300_4, label=labels_300[3])
ax2.text(0.75,1/3,f'M max = {mag_max_300:.0f} A/m\n{len(conc_res_300)} muestras',bbox=dict(facecolor='tab:blue', alpha=0.5),transform=ax2.transAxes,va='center',ha='center')


ax2.set_title('300 kHz', fontsize=14)

ax1.set_ylabel('M (A/m)')
for a in [ax1,ax2]:
    a.grid()
    a.legend(ncol=1)
    a.set_xlabel('H (A/m)')
plt.suptitle('Paramagneto (Dy$_2$O$_3$) - 12/12/24')
plt.savefig('ciclos_100_300.png', dpi=400, facecolor='w')
plt.savefig('Paramagneto_100_300_241212.png',dpi=300)
plt.show()

# %%
