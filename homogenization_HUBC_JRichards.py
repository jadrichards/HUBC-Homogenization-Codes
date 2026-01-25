#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Joshua Richards
Email: jrichar2@umbc.edu
Github User: jadrichards
Paper: J.A.D.Richards et al 2026 (Paper in Review. Link will be updated upon paper approval)

Program Description:
    This the main code to apply all correction to the Howard University Ozonesonde Data Record and produces final 
    homogenized records in SHADOZV06 format

PLEASE NOTE: This program is fit for the Howard University data set. Even though this uses 
recommendations from Smit et al 2021,for homogenizing other station record you will need to
change the applied corrections based on what metadata (information from ozonesonde prep sheets) your station has
    
"""

import time
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #optiona;, This removes any warning that wont effect the results

#CHANGE LATER JOSHUA
from homogenization_functions_JRichards import absorption_efficiency, stoichmetry_conversion, conversion_efficiency, \
    background_correction,pumptemp_corr, currenttopo3, pf_groundcorrection, calculate_cph, pumpflow_efficiency,return_phipcor, \
    RS_pressurecorrection, o3_integrate, o3tocurrent, ComputeCef


#%%
#Function Parking Lot


def meanVal(df, name, emptyVal = -9999):
    '''
    df = dataframe
    name = name of df column
    emptyVal = empty value indicator
    
    Returns:
        the mean value fo column and skipping over bad values
    '''
    tot = 0
    count = 0
    for i in range(len(df[name][:])):
        if df[name][i] != emptyVal:
            tot = tot + df[name][i]
            count = count  + 1
    mean = tot/count
    return mean

def totalcolumnintegration(pressure, O3P):
    '''
    Parameters
    ----------
    pressure : pressure values
    O3P : Ozone partial pressure values

    Returns
    -------
    o3_du : returns array of o3 in du 
    tco : returns value of total column ozone

    '''
    convDU = 1000
    o3_du = [0]
    t_o0 = 0
    tco = 0
    for i in range(1,len(pressure)):
        p0 = pressure[i-1]
        p1 = pressure[i]
        o3pp0 = O3P[i-1] * 10
        o3pp1 = O3P[i] * 10
        at_cm = t_o0 + (np.log10(p0/p1) * (o3pp0+o3pp1))/1100.0
        tco = at_cm * convDU
        o3_du.append(tco)
        t_o0 = at_cm
    return o3_du, tco 

def addmeans(month, var, df):
    '''
    Parameters
    ----------
    month : number of month (01 = jan)
    var : name of df column
    df : dataframe with values

    Returns
    -------
    df : returns updated dataframe with average value for whatever period of time

    '''
    vals = []
    i = 0
    while i != len(df[var]):
        if str(df["Date"][i])[4:6] == month and df[var][i] != -9999:
            vals.append(df[var][i])
        i = i + 1
    mean = np.mean(vals)
    i = 0
    while i != len(df[var]):
        if str(df["Date"][i])[4:6] == month and df[var][i] == -9999:
            df.loc[i, var] = mean
        i = i + 1
    return df


    
def calc_o3current(ib, tp, pfe, pfr, o3pp):
    '''
    Parameters
    ----------
    ib : background current
    tp : pump temperature
    pfe : pump flow efficiency
    pfr : pump flow rate
    o3pp : ozone partial pressure

    Returns
    -------
    y : calculates the cell current from the other variables in partial pressure equation
    '''
    tp = tp + 273.15
    y = ((o3pp*pfe*pfr)/(tp*4.307*10**-2))+ib
    return y

def extract_filename(file_path):
    '''
    Parameters
    ----------
    file_path : full string path
    Returns
    -------
    filename2[0] : returns just the file name with out full address
    '''
    filename = os.path.basename(file_path)
    filename2 = os.path.splitext(filename)
    return filename2[0]

def grab_Year(input_string):
    '''
    Parameters
    ----------
    input_string : some string name

    Returns
    -------
    returns the year as string
    '''
    parts = input_string.split('_')
    return parts[-2][0:4]

def convert_to_float(val):
    '''
    Parameters
    ----------
    val : some number like string/int/float

    Returns
    -------
    converts value into float or if its not a float just returns -9999
    '''
    try:
        return float(val)
    except ValueError:
        return -9999

def replace_9999(value):
    '''
    Parameters
    ----------
    value : some value

    Returns
    -------
    takes any values that's 9999 and replaces it with 0
    '''
    return 0 if value == 9999.00 else value

def replace_negative(value):
    '''
    Parameters
    ----------
    value : some value

    Returns
    -------
    returns zero if its negative
    '''
    return 0 if value < 0 else value

def replace_ib(value):
    '''
    Parameters
    ----------
    value : some value

    Returns
    -------
    returns default background current which is 0.04

    '''
    return 0.04

def find_check_value(df, a, b, c):
    '''
    Parameters
    ----------
    df : some data file
    a : value of lab pressure
    b : value of lab temperature
    c : value of lab relative humidity

    Returns
    -------
    If that row has a flowrate correction it returns that value, otherwise it will return a -9999 value
    '''
    row = df[(df['Plab'] == a) & (df['LabT'] == b) & (df['LabRH'] == c)]
    if not row.empty:
        return row.iloc[0]['Flowrate correction']
    else:
        return -9999
    
def flowrateCal(t,rh,p):
    '''
    Parameters
    ----------
    t : Temp in Celcius
    rh : Relative Humidty in %
    p : Pressure in hPa

    Returns
    -------
    flowrate correction
    '''
    v1 = ((7.5*t)/(t+237.3))+0.7858
    v2 = 10**v1
    v3 = 1-(rh/100)
    v4 = v2 * v3
    v5 = p - v4
    v6 = v5/p
    correction = (1-v6)*100
    return correction

def replace_values(matrix, target_value=-9000, replacement_value=9000.00):
    '''
    Parameters
    ----------
    matrix : takes in a matrix (probably the matrix right before making the final files)
    target_value : some number that shouldnt exist in your data
    replacement_value : replaces that value with 9000 which skysonde likes

    Returns
    -------
    matrix : returns data with missing values with the number of 9000

    '''
    matrix[matrix < target_value] = replacement_value
    return matrix

def check9999(value):
    '''
    Parameters
    ----------
    value : some value

    Returns
    -------
    if its a float of -9999 or -999.99 we convert it to 9000 which is the actual missing value indicator

    '''
    try:
        val = float(value)
    except:
        return value
    if val == -9999.99 or val == -9999.0:
        return 9000
    else:
        return value

def neighbors(array):
    '''
    array = an array of index locations
    
    Returns all indexes of values that have neighbors but not the end points
    '''
    buddies = []
    i = 1
    while i != len(array):
        if array[i]-array[i-1] == 1:
            if (i+1) != len(array):
                if array[i+1]-array[i] == 1:
                    buddies.append(i)
        i = i + 1
    return buddies

def repeated(array):
    '''
    Parameters
    ----------
    array : an array

    Returns
    -------
    A list of indexs of values that were repeated wtihout counting the first repeat

    '''
    indexes = []
    seen = {}
    
    for index,value in enumerate(array):
        if value in seen:
            indexes.append(index)
        else:
            seen[value] = index
    
    return indexes
    
def removes_at_index(removalArrays, indexes, keep = 0):
    '''
    removalArrays : LIST of arrays to remove the indexes
    indexes : the indexes to remove
    keep: condition to know if we want to keep the removed indexes
    '''
    kicked = []
    for i in sorted(indexes,reverse=True):
        #This will keep the code from crashing out. Removing higher indexes first prevents the lower ones
        #from changing their order
        for array in removalArrays:
            if i < len(array):
                if keep != 0:
                    kicked.append(array[i])
                array.pop(i)
    if keep != 0:
        return kicked
    
def tpumpDifference(df, name):
    '''
    df : dataframe
    name : string; is the column we are looking for 
    Returns True if the distance between any two adjacant values are greater than 3
    
    '''
    data = df[name]
    difference = data.diff().abs()
    return (difference > 3).any()


COL_SPEC = [
    (2,   6,   4,  "{:>4d}",     "Time sec (integer)"),
    (7,   14,  7,  "{:>7.2f}",   "Press hPa"),
    (17,  23,  6,  "{:>6.3f}",   "GeopAlt km"),
    (25,  31,  6,  "{:>6.2f}",   "Temp C"),
    (33,  38,  5,  "{:>5.1f}",   "RH %"),
    (41,  48,  7,  "{:>7.4f}",   "O3_mPa"),
    (51,  57,  6,  "{:>6.4f}",   "O3_ppmv"),
    (58,  65,  7,  "{:>7.2f}",   "O3_DU"),         
    (67,  73,  6,  "{:>6.2f}",   "Wind_Dir deg"),
    (76,  81,  5,  "{:>5.2f}",   "Wind_Spd m/s"),
    (84,  89,  5,  "{:>5.2f}",   "TPump C"),
    (93,  99,  6,  "{:>6.4f}",   "O3CellI uA"),
    (102, 110, 8,  "{:>8.5f}",   "GPS_Lat deg"),
    (112, 121, 9,  "{:>9.5f}",   "GPS_Lon deg"),
    (124, 130, 6,  "{:>6.3f}",   "GPS_Alt km"),
]

def _fw_line(values):
    """
    Build one fixed-width data line using COL_SPEC.
    `values` must be a list/tuple of 15 values in the target order.
    """
    line = [" "] * 131  
    for idx, (start, end, width, fmt, _) in enumerate(COL_SPEC):
        v = values[idx]
        if v is None or (isinstance(v, float) and (v != v)):
            s = " " * width
        else:
            if idx == 0:  
                try:
                    v = int(round(float(v)))
                except Exception:
                    v = 0
            if v == 9000:
                fmt = "{:>4.0f}"
                width = 4
            s = fmt.format(v)
            if len(s) > width:  # overflow safety
                s = s[-width:]
        for i, ch in enumerate(s):
            pos = start + i
            if 0 <= pos < len(line):
                line[pos] = ch
    return "".join(line)

def write_shadoz_v06(fp, header_info, data_mat):
    """
    Write the header and data in SHADOZ V06 format with exact spacing.
    `data_mat` must have 15 columns in this order:
      TimeSec, Press, GeopAlt, Temp, RH, O3_mPa, O3_ppmv, O3_DU,
      Wind_Dir, Wind_Spd, TPump, O3CellI, GPS_Lat, GPS_Lon, GPS_Alt
    """
    # ---- HEADER (adjust strings as needed) ----
    fp.write("36\n")
    fp.write("NASA/GSFC/Archive\n")
    fp.write(f"Wallops Principal Investigator    : {header_info.get('pi', 'Ryan M. Stauffer (NASA/Goddard Space Flight Center, USA; ryan.m.stauffer@nasa.gov)')}\n")
    fp.write(f"Wallops Archive Curator           : {header_info.get('curator', 'Debra E Kollonige (SSAI affiliated with NASA/GSFC, USA)')}\n")
    fp.write("SHADOZ Version                    : 06\n")
    fp.write(f"Reprocessed                       : {header_info.get('reprocessed', 'Yes')}\n")
    fp.write(f"SHADOZ format data created        : {header_info.get('created', datetime.now().strftime('%a %b %d %H:%M:%S %Y'))}\n")
    fp.write(f"STATION                           : {header_info.get('station', 'Wallops Is, VA')}\n")
    fp.write(f"Station Principal Investigator(s) : {header_info.get('pi_short', 'Ryan M. Stauffer (NASA/Goddard Space Flight Center, USA)')}\n")
    fp.write(f"Latitude (deg)                    : {header_info.get('lat', '37.93')}\n")
    fp.write(f"Longitude (deg)                   : {header_info.get('lon', '-75.47')}\n")
    fp.write(f"Elevation (m)                     : {header_info.get('elev_m', '11.00')}\n")
    fp.write(f"Launch Date                       : {header_info.get('launch_date', 'YYYYMMDD')}\n")
    fp.write(f"Launch Time (UT)                  : {header_info.get('launch_time', 'HH:MM:SS')}\n")
    fp.write(f"Burst Pressure (hPa)              : {header_info.get('pburst', '')}\n")
    fp.write(f"Integrated O3 to end of data (DU) : {header_info.get('o3_du_total', '')}\n")
    fp.write(f"Radiosonde                        : {header_info.get('rstype','')}\n")
    fp.write(f"Radiosonde Serial Number          : {header_info.get('rsnum', '')}\n")
    fp.write(f"Geom. Press. Offset Correction    : {header_info.get('goffset', '')}\n")
    fp.write(f"Pressure sensor offset (hPa)      : {header_info.get('offset', '')}\n")
    fp.write(f"Ozonesonde Instrument             : {header_info.get('sondeNum','')}\n")
    fp.write(f"KI Solution                       : {header_info.get('solution','')}\n")
    fp.write("Cathode Soln Volume (cc)          : 3.0\n")
    fp.write("Cathode Soln Volume Correction    : No\n")
    fp.write(f"Pump flowrate (sec/100ml)         : {header_info.get('flowrate','')}\n")
    fp.write(f"Flowrate Correction (%)           : {header_info.get('flcorr','')}\n")
    fp.write("Pump Temperature Correction       : Yes\n")
    fp.write(f"Applied background current (uA)   : {header_info.get('bkgCurrent','')}\n")
    fp.write(f"Applied pump efficiency factors   : {header_info.get('pumpeff', '')}\n")
    fp.write(f"Transfer Function                 : {header_info.get('tfunc', '')}\n")
    fp.write("Missing or bad values             : 9000\n")
    fp.write(f"Comment : {header_info.get('comment1', 'Reprocessing reference: Witte et al. [https://doi.org/2018JD030098]')}\n")
    fp.write(f"Comment : {header_info.get('comment2', 'Bandy and Torres (1982) T-based only flowrate correction applied.')}\n")
    fp.write("Comment : \n")

    # Updated column headers with O3_DU added
    fp.write("Time   Press    GeopAlt   Temp    RH     O3_mPa   O3_ppmv  O3_DU  Wind_Dir Wind_Spd  TPump  O3CellI  GPS_Lat    GPS_Lon   GPS_Alt\n")
    fp.write("sec    hPa      km        C       %      mPa      ppmv     DU     deg      m/s       C      uA       deg        deg       km     \n")

    # ---- DATA ----
    for row in data_mat:
        vals = []
        for x in row[:15]:
            if isinstance(x, (int, float)) and abs(float(x) - 9000.0) < 1e-12:
                vals.append(9000)  # print as blanks
            else:
                vals.append(x)
        fp.write(_fw_line(vals) + "\n")
        
def grab_file(filename, paths):
    # print(printpaths)
    for path in paths:
        # print(filename, str(path)[-29:-14])
        if filename == str(path)[-29:-14]:
            return path
        
    print("No Matching 2006 Name")
    return None

k = 273.15 #Kelvin
#%%

#Establishing where all of my files are located.

#Depending on if I am on my windows computer or mac
i = 0
while i != 1:
    check = input("Is this Windows (1) or Mac (2): ")
    try:
        val = int(check)
        if val == 1:
            addy = '/Users/jrich/iCloudDrive/Documents/Nasa Spring 2023 Proj/'
            i = 1
        elif val == 2:
            addy = '/Users/jushuwa/Documents/Nasa Spring 2023 Proj/'
            i = 1
        else:
            print("Input 1 or 2")
    except:
        print("Input 1 or 2")


dir_path = addy+'HUBV Homogenization/PreHomogenized Files/' #Location of all prehomogenized files
dfmeta = pd.read_csv(addy+'HUBV_metadata_liveedit copy.csv', header=0) #Location of Metafile
filehome = addy + 'HUBV Homogenization/PostHom_PreSkySonde_03142025/' #Location of Folder where I will save all outputs

datpaths = [x for x in Path(dir_path).rglob("*.DAT")] #List of all prehomogenized paths
profiles2006 = [x for x in Path(addy+'HUBV Homogenization/PreHomogenized Files_2006/').rglob("*.DAT")] #All 2006 prehomogenized profiles that were run through skysonde first to fix pump efficiencies (See paper)
# %%

start_time = time.time() #Checking how long this code runs for

pd.options.mode.chained_assignment = None  # default='warn'

#Formatting metafile csv to make sure all need values are filled...

#   Adds Climatalogical means for profiles that are missing these 
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
for x in months:
    dfmeta = addmeans(x, "LabP", dfmeta)
    dfmeta = addmeans(x, "LabT", dfmeta)
    dfmeta = addmeans(x, "LabRH", dfmeta)

dfmeta["Date1"] = [pd.to_datetime(str(dfmeta['Date'][i])+' '+str(dfmeta['GMT'][i]), format='%Y%m%d %H:%M:%S') for i in np.arange(len(dfmeta['GMT']))]
dfmeta["Date2"] = [pd.to_datetime(str(dfmeta['Date'][i]), format='%Y%m%d') for i in np.arange(len(dfmeta['GMT']))]
dfmeta['Pground'] = dfmeta['LabP']
dfmeta = calculate_cph(dfmeta)
dfmeta['unc_cPH'] = dfmeta['cPH'].std()
dfmeta['unc_cPL'] = dfmeta['cPL'].std()

#Sanity checks
succeeded = 0  
fails = 0
errorfiles = []
total = len(datpaths)
counter = 0
dates = []
diffTest = 0
diffNames = []
pumptempchanged = []
filenum = 0


# Main Loop to homogenize all files
for filename in datpaths:
    
    filenum = filenum + 1
    totdat = len(datpaths)
    name = extract_filename(filename)[-25:-10]
    print("\nWorking on",name)
    date_ymdt = str(filename)[-29:-14]
    year = str(date_ymdt[0:4])
    
    try:
        
        if year == "2006": #This is ONLY for 2006 years where the location of the files were in another location
            realfilename = grab_file(name, profiles2006)
            df_file = pd.read_csv(realfilename, skiprows=(18))
            df_file.columns = ['Time[min]','Press[hpa]','Alt[km]','Temp[deg C]','RH[%%]','O3 Cell I[uA]','O3 P[mPa]','O3 Mr[ppmv]','T Pump[deg C]','GPS lat[deg]','GPS lon[deg]','GPS alt[km]','Wind Speed[m/s]','Wind Dir[deg]']
            preskyfile =  pd.read_csv(filename, skiprows=(18))
            preskyfile.columns = ['Time[min]','Press[hpa]','Alt[km]','Temp[deg C]','RH[%%]','O3 Cell I[uA]','O3 P[mPa]','O3 Mr[ppmv]','T Pump[deg C]','GPS lat[deg]','GPS lon[deg]','GPS alt[km]','Wind Speed[m/s]','Wind Dir[deg]']
        else:
            df_file = pd.read_csv(filename, skiprows=(18))
            df_file.columns = ['Time[min]','Press[hpa]','Alt[km]','Temp[deg C]','RH[%%]','O3 Cell I[uA]','O3 P[mPa]','O3 Mr[ppmv]','T Pump[deg C]','GPS lat[deg]','GPS lon[deg]','GPS alt[km]','Wind Speed[m/s]','Wind Dir[deg]']
        
        counter = counter + 1
        month = str(date_ymdt[4:6])
        day = str(date_ymdt[6:8])
        hour = str(date_ymdt[9:11])
        mins = str(date_ymdt[11:13])
        
        try:
            secs = str(date_ymdt[13:15])
        except:
            secs = "00"
        
        mtestGMT = hour+':'+mins+':'+secs
        mtestDATE = int(year+month+day)
        date = day + '-' + month + '-' + year
        
        #Finds the matching row in the metafile to use for the homogization of the current file (df_file)
        mask = (dfmeta["GMT"].str[:5] == mtestGMT[:5]) & (dfmeta["Date"] == mtestDATE)
        matching_indices = dfmeta[mask].index
        if not matching_indices.empty:
            tag = matching_indices[0]
        else:
            tag = -1
            print("didn't find a match for", name)
            continue
        TAG = tag
        
        
        #Cleans up the prehomogenized data to remove any bad values (ie. negative measured sensor currents)
        min_index = df_file["Press[hpa]"].idxmin()
        df_file = df_file.loc[:min_index]
        test = len(df_file["Press[hpa]"])
        flagColumns = ['Press[hpa]','Alt[km]','Temp[deg C]','RH[%%]','O3 Cell I[uA]','O3 P[mPa]']
        for column in flagColumns:
            df_file[column] = df_file[column].apply(convert_to_float)
        df_file['RH[%%]'] = df_file['RH[%%]'].apply(replace_9999)
        df_file = df_file[~((df_file[flagColumns] <= -8999).any(axis=1) | (df_file[flagColumns] > 5000).any(axis=1))]
        print(test - len(df_file["Press[hpa]"]), "bad values were scrubbed")
        dropped = test - len(df_file["Press[hpa]"])
        df_file = df_file.reset_index(drop = True)
        negCheck = ['O3 Cell I[uA]','O3 P[mPa]','O3 Mr[ppmv]']
        for column in negCheck:
            df_file[column] = df_file[column].apply(replace_negative)
        
        
        #Formatting files with some default values
        df_file['Eta'] = 1
        df_file['dPhip'] = 0.02 #assumption from 8.4 recommendations
        df_file['unc_cPH'] = dfmeta.at[dfmeta.first_valid_index(), 'unc_cPH']
        df_file['unc_cPL'] = dfmeta.at[dfmeta.first_valid_index(), 'unc_cPL']
        df_file['Tpump'] = df_file['T Pump[deg C]'] + k
        labt = dfmeta['LabT'][TAG]
        df_file['LabT'] = labt
        
        #----------ETA Correction (Conversion Efficiency)------------ 
        # 1) getting absorption efficiency
        df_file['alpha_o3'], df_file['unc_alpha_o3'] = absorption_efficiency(df_file, 'Press[hpa]', 3.0)
        
        # 2) Fixing naming convension
        if dfmeta['SensorType'][TAG] == 'SPC': pumpflowtable = 'komhyr_86'
        if dfmeta['SensorType'][TAG] == 'DMT-Z' or dfmeta['SensorType'][TAG] == 'ENSCI': 
            pumpflowtable = 'komhyr_95'
            dfmeta.loc[TAG, 'SensorType'] = 'DMT-Z'
        
        # 3) Applying Transfer Function
        transferflag = 0
        if dfmeta['SensorType'][TAG] == 'DMT-Z':
            df_file['stoich'], df_file['unc_stoich'] = stoichmetry_conversion(df_file, 'Press[hpa]', dfmeta['SensorType'][TAG],
                                                                '5','ENSCI05')
        else:
            df_file['stoich'], df_file['unc_stoich'] = stoichmetry_conversion(df_file, 'Press[hpa]', dfmeta['SensorType'][TAG],
                                                                '5','SPC10')
            transferflag = 1
        
        #4) FINAL ETA correction
        df_file['eta_c'], df_file['unc_eta_c'] = conversion_efficiency(df_file, 'alpha_o3', 'unc_alpha_o3', 'stoich', 'unc_stoich')
        
        
        
        #---------------Background correction-----------#
        i = 0
        while i != len(dfmeta["GMT"]):
            #formatting, if there is a recorded ib2 we use ib0
            try:
                dfmeta.loc[i,"IB2"] = float(dfmeta["IB2"][i])
                if dfmeta["IB2"][i]  == -9999:
                    dfmeta.loc[i,"IB2"]  = dfmeta["IB0"][i]
            except:
                dfmeta.loc[i,"IB2"]  = dfmeta["IB0"][i]
                dfmeta.loc[i,"IB2"]  = float(dfmeta["IB2"][i])
            i = i + 1
        
        IBGsplit = 0
        #Final BKG Correction
        df_file['iBc'], df_file['unc_iBc'] = background_correction(df_file, dfmeta, TAG, 'IB2', IBGsplit)
        df_file['iB2'] = dfmeta['IB2'][tag]

        
        #Radiosonde (RS80) Pressure Offset Correction
        rsmodel = dfmeta['RadiosondeType'][tag]
        if "RS80" in rsmodel:
            df_file['Crs'], df_file['unc_Crs'] = RS_pressurecorrection(df_file, 'Alt[km]', rsmodel)
            #Final Pressure Offset Correction Applied
            df_file['Press[hpa]'] = df_file['Press[hpa]'] - df_file['Crs']
        
        
        #Ground Correction with Humidification correction
        df_file['Phip'] = 100 / dfmeta['FlowRate'][tag]
        df_file['Phip_ground'], df_file['unc_Phip_ground'] = pf_groundcorrection(df_file, dfmeta, 'Phip', 'dPhip', 'LabT', 'LabP', 'LabRH', True)
        df_file['Cpf'], df_file['unc_Cpf'] = pumpflow_efficiency(df_file, 'Press[hpa]', pumpflowtable, ' ') #Since pump efficiencies were the recommended values we didnt change them
        df_file['Phip_cor'], df_file['unc_Phip_cor'] = return_phipcor(df_file, 'Phip_ground', 'unc_Phip_ground', 'Cpf', 'unc_Cpf')
        if dfmeta['SensorType'][TAG] == 'DMT-Z':
            dfmeta.loc[TAG, 'SensorType'] = 'ENSCI'
        
        #--------Pump Temperature Correction------#
        string_pump_location = 'InternalPump'
        #Pump Temperature Correction
        df_file['unc_Tpump'] = 0.5 #all files I work with have pressure included
        df_file['Tpump_cor'], df_file['unc_Tpump_cor'] = pumptemp_corr(df_file, string_pump_location, 'Tpump', 'unc_Tpump', 'Press[hpa]')
        df_file['Tpump_ori'] = df_file['Tpump_cor']
        
        #Fixing Vaisala Pump Temperature Software Bug (This part is long)
        flag = 0
        if tpumpDifference(df_file, 'Tpump_cor'):
            diffTest = diffTest + 1
            name = extract_filename(filename)[-25:-10]
            diffNames.append(name)
            skipFiles = ["20220720_185414","20080718_060700"]
            if name not in skipFiles:
                tpump = df_file['Tpump_cor']
                h = df_file['Alt[km]']
                
                pumptempchanged.append(filename)
                
                jumpValue_tpump = []
                jumpin_tp = []
                jumpin_h = []
                jumpValue_h = []
                i = 1
                while i != len(tpump):
                    curr = tpump[i]
                    prev = tpump[i-1]
                    check = abs(curr-prev)
                    if check > 1:
                        jumpValue_tpump.append(tpump[i-1])
                        jumpValue_tpump.append(tpump[i])
                        jumpValue_h.append(h[i-1])
                        jumpValue_h.append(h[i])
                        jumpin_tp.append(i-1)
                        jumpin_tp.append(i)
                        jumpin_h.append(i-1)
                        jumpin_h.append(i)
                    i = i + 1

                removes_at_index([jumpValue_tpump,jumpin_tp,jumpin_h,jumpValue_h], repeated(jumpin_h))

                #Isolates the actual jumps and ignores the inbetween for now...
                buddies = neighbors(jumpin_h)
                removedIndexes = removes_at_index([jumpin_h], neighbors(jumpin_h), 1)
                removes_at_index([jumpValue_tpump,jumpin_tp,jumpValue_h], buddies)
                
                #There are 3 scenarios
                if len(jumpin_h) == 4:
                    # print("4 scenario")
                    midpoint = (jumpValue_tpump[0] + jumpValue_tpump[1])/2
                    inbetween = np.arange(jumpin_h[1]+1,jumpin_h[2]+1,1)
                    
                    newT = []
                    for x in inbetween:
                        change = (tpump[x]-tpump[x-1])*-1
                        newT.append(change)
                    
                    tpump[jumpin_h[1]] = tpump[jumpin_h[1]] + 2*abs(midpoint-tpump[jumpin_h[1]])
                    
                    for i in range(len(inbetween)):
                        if i == 0:
                            tpump[inbetween[i]] = newT[i] + tpump[jumpin_h[1]]
                        else:
                            tpump[inbetween[i]] = newT[i] + tpump[inbetween[i-1]]
                    
                    #This following if statement runs on an assumption that before and after the jumps the data has the same value
                    if tpump[jumpin_h[2]] != tpump[jumpin_h[3]]:
                        
                        shift = tpump[jumpin_h[2]]-tpump[jumpin_h[3]]
                        for i in np.arange(jumpin_h[3],len(tpump),1):
                            tpump[i] = tpump[i] + shift
                  
                elif len(jumpin_h) == 5:
                    # print("5 scenario")
                    #Removes that weird one and treats it like a 4 instead
                    removes_at_index([jumpValue_tpump,jumpin_tp,jumpin_h,jumpValue_h], [2])
                    
                    midpoint = (jumpValue_tpump[0] + jumpValue_tpump[1])/2
                    inbetween = np.arange(jumpin_h[1]+1,jumpin_h[2]+1,1)
                    
                    newT = []
                    for x in inbetween:
                        change = (tpump[x]-tpump[x-1])*-1
                        newT.append(change)
                    
                    tpump[jumpin_h[1]] = tpump[jumpin_h[1]] + 2*abs(midpoint-tpump[jumpin_h[1]])
                    
                    for i in range(len(inbetween)):
                        if i == 0:
                            tpump[inbetween[i]] = newT[i] + tpump[jumpin_h[1]]
                        else:
                            tpump[inbetween[i]] = newT[i] + tpump[inbetween[i-1]]
                    
                    #This following if statement runs on an assumption that before and after the jumps the data has the same value
                    if tpump[jumpin_h[2]] != tpump[jumpin_h[3]]:
                        
                        shift = tpump[jumpin_h[2]]-tpump[jumpin_h[3]]
                        for i in np.arange(jumpin_h[3],len(tpump),1):
                            tpump[i] = tpump[i] + shift
                
                elif len(jumpin_h) == 6:
                    # print("6 scenario")
                    midpoint1 = (jumpValue_tpump[0] + jumpValue_tpump[1])/2
                    
                    inbetween = np.arange(jumpin_h[1]+1,jumpin_h[2]+1,1)
                    
                    newT = []
                    
                    for x in inbetween:
                        change = (tpump[x]-tpump[x-1])*-1
                        newT.append(change)
                    
                    tpump[jumpin_h[1]] = tpump[jumpin_h[1]] + 2*abs(midpoint1-tpump[jumpin_h[1]])
                    
                    for i in range(len(inbetween)):
                        if i == 0:
                            tpump[inbetween[i]] = newT[i] + tpump[jumpin_h[1]]
                        else:
                            tpump[inbetween[i]] = newT[i] + tpump[inbetween[i-1]]
                
                    midpoint2 = (jumpValue_tpump[4] + jumpValue_tpump[5])/2
                    inbetween = np.arange(jumpin_h[5]+1,len(tpump),1)
                    
                    newT = []
                    
                    for x in inbetween:
                        change = (tpump[x]-tpump[x-1])*-1
                        newT.append(change)
                    
                    tpump[jumpin_h[5]] = tpump[jumpin_h[5]] + 2*abs(midpoint2-tpump[jumpin_h[5]])
                    
                    for i in range(len(inbetween)):
                        if i == 0:
                            tpump[inbetween[i]] = newT[i] + tpump[jumpin_h[5]]
                        else:
                            tpump[inbetween[i]] = newT[i] + tpump[inbetween[i-1]]
                
                    #This part runs on an assumption that before and after the jumps the data has the same value
                    if tpump[jumpin_h[2]] != tpump[jumpin_h[3]]:
                        
                        shift = tpump[jumpin_h[2]]-tpump[jumpin_h[3]]
                        for i in np.arange(jumpin_h[3],len(tpump),1):
                            tpump[i] = tpump[i] + shift
                    
                elif len(jumpin_h) == 8:
                    # print("8 scenario")
                    midpoint1 = (jumpValue_tpump[0] + jumpValue_tpump[1])/2
                    
                    inbetween = np.arange(jumpin_h[1]+1,jumpin_h[2]+1,1)
                    
                    newT = []
                    
                    for x in inbetween:
                        change = (tpump[x]-tpump[x-1])*-1
                        newT.append(change)
                    
                    tpump[jumpin_h[1]] = tpump[jumpin_h[1]] + 2*abs(midpoint1-tpump[jumpin_h[1]])
                    
                    for i in range(len(inbetween)):
                        if i == 0:
                            tpump[inbetween[i]] = newT[i] + tpump[jumpin_h[1]]
                        else:
                            tpump[inbetween[i]] = newT[i] + tpump[inbetween[i-1]]
                
                    midpoint2 = (jumpValue_tpump[4] + jumpValue_tpump[5])/2
                    inbetween = np.arange(jumpin_h[5]+1,jumpin_h[6]+1,1)
                    
                    newT = []
                    
                    for x in inbetween:
                        change = (tpump[x]-tpump[x-1])*-1
                        newT.append(change)
                    
                    tpump[jumpin_h[5]] = tpump[jumpin_h[5]] + 2*abs(midpoint2-tpump[jumpin_h[5]])
                    
                    for i in range(len(inbetween)):
                        if i == 0:
                            tpump[inbetween[i]] = newT[i] + tpump[jumpin_h[5]]
                        else:
                            tpump[inbetween[i]] = newT[i] + tpump[inbetween[i-1]]
                            
                elif len(jumpin_h) == 2:
                    # print("2 scenario")
                    if jumpValue_tpump[1] > jumpValue_tpump[0]:
                        midpoint = (jumpValue_tpump[0] + jumpValue_tpump[1])/2
                        inbetween = np.arange(jumpin_h[0],0,-1)
                        
                        newT = []
                        
                        for x in inbetween:
                            change = (tpump[x]-tpump[x-1])
                            newT.append(change)
                        newT.append(tpump[1]-tpump[0])
                        
                        tpump[jumpin_h[0]] = tpump[jumpin_h[0]] + 2*abs(midpoint-tpump[jumpin_h[0]])
                        
                        for i in range(len(inbetween)):
                            if i == 0:
                                tpump[inbetween[i]] = newT[i] + tpump[jumpin_h[0]]
                            else:
                                tpump[inbetween[i]] = newT[i] + tpump[inbetween[i-1]]
                        
                        tpump[0] = newT[-1] + tpump[1]
                        
                    else:
                        # print("Not sure about this part yet. Havent tested it")
                        midpoint = (jumpValue_tpump[0] + jumpValue_tpump[1])/2
                        inbetween = np.arange(jumpin_h[1]+1,len(tpump),1)
                        
                        newT = []
                        
                        for x in inbetween:
                            change = (tpump[x]-tpump[x-1])*-1
                            newT.append(change)
                        
                        tpump[jumpin_h[1]] = tpump[jumpin_h[1]] + 2*abs(midpoint-tpump[jumpin_h[1]])
                        
                        for i in range(len(inbetween)):
                            if i == 0:
                                tpump[inbetween[i]] = newT[i] + tpump[jumpin_h[1]]
                            else:
                                tpump[inbetween[i]] = newT[i] + tpump[inbetween[i-1]]
                else:
                    print("I dont have a solution for this type yet")
                    flag = 1
                
                df_file['Tpump_cor'] = tpump
                df_file = df_file.drop(index = removedIndexes)
                df_file = df_file.reset_index(drop = True)
        
        
        if flag == 1:
            continue
        
        # Applying corrections
        df_file['O3_nc'] = currenttopo3(df_file, 'O3 Cell I[uA]', 'Tpump', 'iB2', 'Eta', 'Phip')
        df_file['O3c'] = currenttopo3(df_file, 'O3 Cell I[uA]', 'Tpump_cor', 'iBc', 'eta_c', 'Phip_cor')

        
        if len(df_file[df_file['O3c'] < 0]) > 0: print('!!!CHECK WHY', filename,'HAS A 0> O_3 value!!!') #Sanity checker

        # uncertainities
        df_file['dI'] = 0
        df_file.loc[df_file['O3 Cell I[uA]'] < 1, 'dI'] = 0.01
        df_file.loc[df_file['O3 Cell I[uA]']>= 1, 'dI'] = 0.01 * df_file.loc[df_file['O3 Cell I[uA]'] > 1, 'O3 Cell I[uA]']
        df_file['dIall'] = (df_file['dI'] ** 2 + df_file['unc_iBc'] ** 2) / (df_file['O3 Cell I[uA]'] - df_file['iBc']) ** 2
        df_file['dEta'] = (df_file['unc_eta_c'] / df_file['eta_c']) ** 2
        df_file['dPhi_cor'] = (df_file['unc_Phip_cor'] / df_file['Phip_cor']) ** 2

        # final uncertainity on O3
        df_file['dO3'] = np.sqrt(df_file['dIall'] + df_file['dEta'] + df_file['dPhi_cor'] + df_file['unc_Tpump_cor'])
        
        
        #Integrated O3 in DU
        o3_du, tco = totalcolumnintegration(df_file['Press[hpa]'], df_file["O3c"])
        
        
        #Craeating the SHADOZV06 Files
        name = extract_filename(filename)[-25:-10]
        flightNum = "HUBV_"+ name[:-2] + dfmeta["GMT"][TAG][-2:] 
        newfilename_de1 = "HUBV_" + name[0:8] + "T" + name[9:11] + '_SHADOZV06.dat'
        path = filehome
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = newfilename_de1 
        
        #Matrix of data
        t1 = [np.around(x*60,2) for x in df_file["Time[min]"]]
        t2 = [np.around(x,2) for x in df_file["Press[hpa]"]]
        t3 = [np.around(x,3) for x in df_file["Alt[km]"]]
        t4 = [np.around(x,2) for x in df_file["Temp[deg C]"]]
        t5 = [np.around(x,1) for x in df_file["RH[%%]"]]
        t6 = [np.around(x,4) for x in df_file["O3c"]]
        val = [x*1e-3 for x in df_file["O3c"]]
        p = np.array(df_file["Press[hpa]"])
        nval = np.divide(val,p*1e2)*1000000
        t7 = np.around(nval,4)
        t8 = [np.around(x,2) for x in o3_du]
        t9 = [np.around(x,2) for x in df_file["Wind Dir[deg]"]]
        t10 = [np.around(x,2) for x in df_file["Wind Speed[m/s]"]]
        t11 = [np.around(x-273.15,2) for x in df_file["Tpump_cor"]]
        t12 = [np.around(x,4) for x in df_file["O3 Cell I[uA]"]]
        t13 = [np.around(x,5) for x in df_file["GPS lon[deg]"]]
        t14 = [np.around(x,5) for x in df_file["GPS lat[deg]"]]
        t15 = [np.around(x,3) for x in df_file["GPS alt[km]"]]
        
        matrix = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15]
        df=pd.DataFrame(matrix)
        mat = df.transpose()
        mat = np.array(mat)
        mat = np.nan_to_num(mat, nan=9000)
        mat = replace_values(mat)
        pburst = str(df_file["Press[hpa]"][len(df_file["Press[hpa]"])-1])
        if dfmeta['SensorType'][TAG] == 'SPC':
            efffactor = 'Komhyr et al., 1986'
        else:
            efffactor = 'Komhyr et al., 1995'
        if transferflag == 1:
            tfunc = 'Yes'
        if transferflag == 0:
            tfunc = 'No'
        bkgCurrent = str(df_file["iBc"][0])
        
        with open(os.path.join(filehome, file_name), 'w', encoding='utf-8', newline='\n') as fp:
            header_info = {
                "pi": "Ryan M. Stauffer (NASA/Goddard Space Flight Center, USA; ryan.m.stauffer@nasa.gov)",
                "curator": "Debra E Kollonige (SSAI affiliated with NASA/GSFC, USA)",
                "reprocessed": "Yes",
                "created": datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
                "station": "Howard University Beltville Campus, USA",
                "pi_short": "Dr. Ricardo K. Sakai (HUBC, USA)",
                "lat": f"{float(39.05634):.2f}",
                "lon": f"{float(-76.8755):.2f}",
                "elev_m": f"{float(0.00):.2f}",
                "launch_date": name[0:8],   # YYYYMMDD
                "launch_time": str(check9999(dfmeta["GMT"][TAG])),   # HH:MM:SS
                "pburst": f"{float(pburst):.2f}",
                "o3_du_total": f"{float(tco):.2f}",
                "rstype": str(dfmeta["RadiosondeType"][TAG]),
                "rsnum": str(check9999(dfmeta["Interface Serial Number"][TAG])),
                "goffset": "No",
                "offset": 0.0,
                "sondeNum": str(check9999(dfmeta["SondeNumber"][TAG])),
                "solution": str(dfmeta["Solution"][TAG]),
                "flowrate": str(dfmeta["FlowRate"][TAG]),
                "flcorr": str(np.around(df_file["cPH"][0]*100,2)),
                "pumpeff": efffactor,
                "tfunc": tfunc,
                "bkgCurrent": bkgCurrent,
                "comment1": "Reprocessing reference: Witte et al. [https://doi.org/2018JD030098]",
                "comment2": "Beltsville Homogenization program (python version) made by Joshua Richards jrich5170@gmail.com",
                }
            write_shadoz_v06(fp, header_info, mat)
            fp.close()
        
        
        print(name + " is done")
        succeeded = succeeded + 1
        print(filenum, "out of",totdat)
        
    except:
        print("\nError1",filename)
        errorfiles.append(filename)
        fails = fails + 1
        print(filenum, "out of",totdat)
    

#Just some sanity check for you later
print('\n',succeeded, "files were homogenized and ", fails,"files caught an error")

print("\nErrorfiles were:")
if len(errorfiles) < 10:
    for x in errorfiles:
        print(x)
elif (len(errorfiles) == 0):
    print("Everything Homogenized")
else:
    print("Your code is cooked (more than 10 error files)")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nScript executed in: {elapsed_time:.4f} seconds or")
print(elapsed_time/60,"mins")

pumptempchangedstr = [str(i)[-29:-14] for i in pumptempchanged]
np.savetxt(addy[:-22]+"Testing_01022025/jumpfiles.txt",pumptempchangedstr,fmt='%s') #saves a list of those files that had to be fixed with the pump temp software bug corrections

