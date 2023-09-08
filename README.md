# Energy-Planning-
This was an assignment to find the best energy system options for one of the course at NTNUa

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:18:12 2023

@author: aayam
"""

import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
from shapely.geometry import LineString
import matplotlib.patches as mpatches
import scipy as sp
import scipy.integrate as integrate

#fixed cost
f_chp = 1000
f_hp = 800
f_bio = 500
f_elb = 60

#energy cost
en_chp = 0.01
en_hp = 0.12
en_bio = 0.01
en_elb = 0.3


#efficiencies 
COP= 3
eff_elb = 0.99
eff_bio = 1
eff_chp = 0.9
af_chp = 0.3
pf_chp = 0.8
eff_car = 0.40

M = 5
a = np.zeros(M)


a[0] = 0 ; a[1] = -0.10; a[2] = -0.20

for j in range (0,3,1):
    eff_elb = 0.99+a[j]
    eff_bio = 1+a[j]
   
    
    O = 5
    b = np.zeros(O)
    c = np.zeros(O)
    b[0] = 0 ; b[1] = -0.02; b[2] = -0.04
    c[0] = 0 ; c[1] = -0.05; c[2] = -0.05
    
    for k in range(0,3,1):
        en_chp = en_chp
        en_hp = en_hp + b[k]
        en_bio = 0.01
        en_elb = en_elb + b[k]
        
         
        N = 5
        h = np.zeros(N)
        I = np.zeros(N) 
        J = np.zeros(N)
        K = np.zeros(N)
     
        h[0] = 100; h[1] =110; h[2] = 121;  h[3] = 133.1;  h[4] = 146.41
        I[0] = 80;  h[1] =88;  I[2] = 96.8; I[3] = 106.48; I[4] = 117.12 
        J[0] = 50;  J[1] =55;  J[2] = 60.5; J[3] = 66.5;   J[4] = 73.2 
        K[0] = 0;   K[1] =0; K[2] = 0; K[3] = 0;   K[4] = 0
    
    
        #first year 
        ele_bio_tc_1 = []
        ele_hp_tc_1 = []
        ele_chp_tc_1 = []
        ele_hp_chp_tc_1 = []
        
        ele_bio_fc_1 = []
        ele_hp_fc_1 = []
        ele_chp_fc_1 = []
        ele_hp_chp_fc_1 = []
        
        ele_bio_oc_1 = []
        ele_hp_oc_1 = []
        ele_chp_oc_1 = []
        ele_hp_chp_oc_1 = []
        
        #Second year
        ele_bio_tc_2 = []
        ele_hp_tc_2 = []
        ele_chp_tc_2 = []
        ele_hp_chp_tc_2 = []
        
        ele_bio_fc_2 = []
        ele_hp_fc_2 = []
        ele_chp_fc_2 = []
        ele_hp_chp_fc_2 = []
        
        ele_bio_oc_2 = []
        ele_hp_oc_2 = []
        ele_chp_oc_2 = []
        ele_hp_chp_oc_2 = []
        
        #Third year
        
        ele_bio_tc_3 = []
        ele_hp_tc_3 = []
        ele_chp_tc_3 = []
        ele_hp_chp_tc_3 = []
        
        ele_bio_fc_3 = []
        ele_hp_fc_3 = []
        ele_chp_fc_3 = []
        ele_hp_chp_fc_3 = []
        
        ele_bio_oc_3 = []
        ele_hp_oc_3 = []
        ele_chp_oc_3 = []
        ele_hp_chp_oc_3 = []
    
        for i in range(0,5,1):
        
        #fixed cost after efficiencies taken into account
        
            f_chp = f_chp + h[i]
            f_hp = f_hp + I[i]
            f_bio = f_bio + J[i]
            f_elb = f_elb + K[i] 
            
        #variable cost after efficiencies taken into account
        
            v_chp = en_chp * af_chp * (1+ pf_chp)/eff_chp
            v_hp = (en_hp/COP)
            v_bio = (en_bio/eff_bio)
            v_elb =(en_elb/eff_elb)
        
        #total specific cost including efficiencies 
        
            t = np.linspace(0,8760,8760)
            cost_chp = (f_chp + v_chp * t)
            cost_hp = (f_hp + v_hp * t)
            cost_bio = (f_bio + v_bio * t)
            cost_elb = (f_elb + v_elb * t)
        
        #Finding point of intersection
        
            bio_elb_x = (f_bio - f_elb )/ (v_elb - v_bio)
            bio_elb_y = (v_bio* bio_elb_x + f_bio)
            elb_hp_x = (f_elb - f_hp )/ (v_hp - v_elb)
            elb_hp_y = (v_elb * elb_hp_x + f_elb)
            elb_chp_x =(f_elb - f_chp)/(v_chp - v_elb)
            elb_chp_y = (v_elb * elb_chp_x + f_elb)
            hp_chp_x = (f_hp - f_chp)/(v_chp - v_hp)
            hp_chp_y = (v_hp * hp_chp_x + f_hp)
        
        #Plotting cost for different system option
        
            plt.figure(figsize=(15,10))
            plt.plot(t,cost_chp , 'r', cost_hp,'g', t,cost_bio,'y', t,cost_elb, 'b')
            plt.xlabel("Time(hrs)")
            plt.ylabel("EURO/KW")
            plt.legend(['Cost of CHP', 'Cost of HP','Cost of bio','Cost of elb'])
            plt.axvline(x=bio_elb_x, color='gray', linestyle='--')
        #plt.axhline(y=bio_elb_y, color='gray', linestyle='--')
            plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
        #plt.axhline(y=elb_hp_y, color='gray', linestyle='--')
            plt.axvline(x=elb_chp_x, color='gray', linestyle='--')
        #plt.axhline(y=elb_chp_y, color='gray', linestyle='--')
            plt.axvline(x=hp_chp_x, color='gray', linestyle='--')
        #plt.axhline(y=hp_chp_y, color='gray', linestyle='--')
        
        
        #plotting point of intersection
            plt.scatter(bio_elb_x, bio_elb_y, color='black')
            plt.annotate('(intersection1',xy=(bio_elb_x, bio_elb_y))
            plt.scatter(elb_hp_x, elb_hp_y, color='black')
            plt.text(bio_elb_x, 1500, 'Electric boiler and Bio boiler', color='green',
                 rotation=90, rotation_mode='anchor')
            plt.annotate('Intersection2',xy=(elb_hp_x, elb_hp_y),xytext=(2600, 800))
            plt.scatter(elb_chp_x, elb_chp_y, color='black')
            plt.text(elb_hp_x, 1500, 'Electric boiler and Heat Pump', color='green',
                 rotation=90, rotation_mode='anchor')
            plt.annotate('Intersection3',xy=(elb_chp_x, elb_chp_y),xytext=(2600, 1150))
            plt.scatter(hp_chp_x, hp_chp_y, color='black')
            plt.text(elb_chp_x, 1500, 'Electric boiler and CHP', color='green',
                 rotation=90, rotation_mode='anchor')
            plt.annotate('Intersection4',xy=(hp_chp_x, hp_chp_y),xytext=(6000, 1200))
            plt.text(hp_chp_x, 1500, 'Heat pump and CHP', color='green',
                 rotation=90, rotation_mode='anchor')
            plt.grid()
            plt.title('Cost Comparison', fontsize=14)
            plt.show()
        
            duration = pd.read_csv('durationcurve.csv')
        
        
            x_axis = duration["SN"]
            y1_axis = duration["2010"]
            y2_axis = duration["2012"]
            y3_axis = duration["2014"]
        
            plt.figure(figsize=(15,10))
            plt.suptitle(' Duration curve', fontsize=14)
        #plt.title("Duration Curve")
        #plt.subplot(2,2,1)
        #plt.plot(x_axis, y1_axis,'b--')
        
        #plt.legend(['Duration curve for 2010'])
        #plt.axvline(x=bio_elb_x, color='gray', linestyle='--')
        #plt.axhline(y=bio_elb_y, color='gray', linestyle='--')
        #plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
        #plt.axhline(y=elb_hp_y, color='gray', linestyle='--')
        #plt.axvline(x=elb_chp_x, color='gray', linestyle='--')
        #plt.axhline(y=elb_chp_y, color='gray', linestyle='--')
        #plt.axvline(x=hp_chp_x, color='gray', linestyle='--')
        #plt.axhline(y=hp_chp_y, color='gray', linestyle='--')
        #plt.grid()
        
        #plt.subplot(2,2,2)
        #plt.plot(x_axis, y2_axis, 'g--')
        #plt.xlabel("Time(hrs)")
        #plt.ylabel("Power(kw)")
        #plt.legend(['Duration curve for 2012'])
        #plt.axvline(x=bio_elb_x, color='gray', linestyle='--')
        #plt.axhline(y=bio_elb_y, color='gray', linestyle='--')
        #plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
        #plt.axhline(y=elb_hp_y, color='gray', linestyle='--')
        #plt.axvline(x=elb_chp_x, color='gray', linestyle='--')
        #plt.axhline(y=elb_chp_y, color='gray', linestyle='--')
        #plt.axvline(x=hp_chp_x, color='gray', linestyle='--')
        #plt.axhline(y=hp_chp_y, color='gray', linestyle='--')
        #plt.grid()
        
        #plt.subplot(2,2,3)
        #plt.plot(x_axis, y3_axis, 'r--')
        #plt.xlabel("Time(hrs)")
        #plt.ylabel("Power(kw)")
        #plt.legend(['Duration curve for 2014'])
        #plt.axvline(x=bio_elb_x, color='gray', linestyle='--')
        #plt.axhline(y=bio_elb_y, color='gray', linestyle='--')
        #plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
        #plt.axhline(y=elb_hp_y, color='gray', linestyle='--')
        #plt.axvline(x=elb_chp_x, color='gray', linestyle='--')
        #plt.axhline(y=elb_chp_y, color='gray', linestyle='--')
        #plt.axvline(x=hp_chp_x, color='gray', linestyle='--')
        #plt.axhline(y=hp_chp_y, color='gray', linestyle='--')
        #plt.grid()
        
        #plt.subplot(2,2,4)
            plt.plot(x_axis, y1_axis, 'b--', x_axis, y2_axis, 'g--', x_axis, y3_axis, 'r--')
            plt.axvline(x=bio_elb_x, color='gray', linestyle='--')
            plt.text(bio_elb_x, 10500, 'Electric boiler and Bio boiler', color='green',
                 rotation=90, rotation_mode='anchor')
            plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
            plt.text(elb_hp_x, 10500, 'Electric boiler and Heat Pump', color='green',
                 rotation=90, rotation_mode='anchor')
            plt.axvline(x=elb_chp_x, color='gray', linestyle='--')
            plt.text(elb_chp_x, 10500, 'Electric boiler and CHP', color='green',
                 rotation=90, rotation_mode='anchor')
            plt.axvline(x=hp_chp_x, color='gray', linestyle='--')
            plt.text(hp_chp_x, 10500, 'Heat pump and CHP', color='green',
                 rotation=90, rotation_mode='anchor')
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            plt.legend(['Duration curve for 2010','Duration curve for 2012','Duration curve for 2014'])
            plt.grid()
        #plt.tight_layout()
        #plt.suptitle('Duration curve', fontsize=14)
         
            plt.show()
        
        
        #Find point of intersection of different solution and the duration curve 
        
        
        #Electric boiler and Bio
        
        #. Bio and electric boiler for 2010
            plt.figure(figsize=(15,10))
            plt.suptitle('Cost using electric boiler and bio boiler during year of operation', fontsize=14)
            plt.subplot(2,2,1)
            plt.plot(x_axis, y1_axis,'b--')
            plt.axvline(x=bio_elb_x, color='gray', linestyle='--')
            intercept1 = np.interp(bio_elb_x, x_axis ,y1_axis)
            print(intercept1)
            plt.scatter(bio_elb_x, intercept1, color='black')
            plt.axhline(y=intercept1, color='gray', linestyle='--')
            plt.annotate('Point of intersection',xy=(bio_elb_x, intercept1))
            point11 = [0, intercept1]
            point21 = [bio_elb_x, intercept1]
            x_values1 = [point11[0], point21[0]]
            y_values1 = [point11[1], point21[1]]
            plt.plot(x_values1, y_values1, 'bo', linestyle="--")
            plt.fill_between(x_axis,y1_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y1_axis,intercept1,where = (x_axis>= 0) & (x_axis <= bio_elb_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='Bio boiler')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2010'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=bio_elb_x))
            area_ele11 = np.trapz(y= np.array(y1_axis)[idx], x=np.array(x_axis)[idx])
            area_t11 = np.trapz(y1_axis, x_axis , dx =1)
            area_bio11 = area_t11 - area_ele11
            print("The total energy requirement 2010 is",{area_t11})
            print("The total energy delivered by electric is",{area_ele11})
            print("The total energy delivered by bio boiler is",{area_bio11})
            plt.grid()
        
        #. Bio and electric boiler for 2012
            plt.subplot(2,2,2)
            plt.plot(x_axis, y2_axis,'b--')
            plt.axvline(x=bio_elb_x, color='gray', linestyle='--')
            intercept2 = np.interp(bio_elb_x, x_axis ,y2_axis)
            print(intercept2)
            plt.scatter(bio_elb_x, intercept2, color='black')
            plt.axhline(y=intercept2, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(bio_elb_x, intercept2))
            point12 = [0, intercept2]
            point22 = [bio_elb_x, intercept2]
            x_values2 = [point12[0], point22[0]]
            y_values2 = [point12[1], point22[1]]
            plt.plot(x_values2, y_values2, 'bo', linestyle="--")
            plt.fill_between(x_axis,y2_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y2_axis,intercept2,where = (x_axis>= 0) & (x_axis <= bio_elb_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='Bio boiler')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2012'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            plt.grid()
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=bio_elb_x))
            area_ele12 = np.trapz(y= np.array(y2_axis)[idx], x=np.array(x_axis)[idx])
            area_t12 = np.trapz(y2_axis, x_axis , dx =1)
            area_bio12 = area_t12 - area_ele12
            print("The total energy requirement 2012 is",{area_t12})
            print("The total energy delivered by electric is",{area_ele12})
            print("The total energy delivered by bio boiler is",{area_bio12})
        
        #. Bio and electric boiler for 2014
            plt.subplot(2,2,3)
            plt.plot(x_axis, y3_axis,'b--')
            plt.axvline(x=bio_elb_x, color='gray', linestyle='--')
            intercept3 = np.interp(bio_elb_x, x_axis ,y3_axis)
            print(intercept3)
            plt.scatter(bio_elb_x, intercept3, color='black')
            plt.axhline(y=intercept3, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(bio_elb_x, intercept3))
            point13 = [0, intercept3]
            point23 = [bio_elb_x, intercept3]
            x_values3 = [point13[0], point23[0]]
            y_values3 = [point13[1], point23[1]]
            plt.plot(x_values3, y_values3, 'bo', linestyle="--")
            plt.fill_between(x_axis,y3_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y3_axis,intercept3,where = (x_axis>= 0) & (x_axis <= bio_elb_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='Bio boiler')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2014'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            plt.grid()
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=bio_elb_x))
            area_ele13 = np.trapz(y= np.array(y3_axis)[idx], x=np.array(x_axis)[idx])
            area_t13 = np.trapz(y3_axis, x_axis , dx =1)
            area_bio13 = area_t13 - area_ele13
            print("The total energy requirement 2014 is",{area_t13})
            print("The total energy delivered by electric is",{area_ele13})
            print("The total energy delivered by bio boiler is",{area_bio13})
            plt.tight_layout()
        
            plt.show()
        
        #Electric boiler and HP 
        
        #HP and electric boiler for 2010
        
            plt.figure(figsize=(15,10))
            plt.suptitle('Cost using electric boiler and heatpump during year of operation', fontsize=14)
            plt.subplot(2,2,1)
            plt.plot(x_axis, y1_axis,'b--')
            plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
            intercept11 = np.interp(elb_hp_x, x_axis ,y1_axis)
            print(intercept11)
            plt.scatter(elb_hp_x, intercept11, color='black')
            plt.axhline(y=intercept11, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_hp_x, intercept11))
            point14 = [0, intercept11]
            point24 = [elb_hp_x, intercept11]
            x_values4 = [point14[0], point24[0]]
            y_values4 = [point14[1], point24[1]]
            plt.plot(x_values4, y_values4, 'bo', linestyle="--")
            plt.fill_between(x_axis,y1_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y1_axis,intercept11,where = (x_axis>= 0) & (x_axis <= elb_hp_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='Heat pump')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2010'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_hp_x))
            area_ele21 = np.trapz(y= np.array(y1_axis)[idx], x=np.array(x_axis)[idx])
            area_t21 = np.trapz(y1_axis, x_axis , dx =1)
            area_hp21 = area_t21 - area_ele21
            print("The total energy requirement 2010 is",{area_t21})
            print("The total energy delivered by electric is",{area_ele21})
            print("The total energy delivered by heat pump is",{area_hp21})
            plt.grid()
        
        #. HP and electric boiler for 2012
            plt.subplot(2,2,2)
            plt.plot(x_axis, y2_axis,'b--')
            plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
            intercept22 = np.interp(elb_hp_x, x_axis ,y2_axis)
            print(intercept22)
            plt.scatter(elb_hp_x, intercept22, color='black')
            plt.axhline(y=intercept22, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_hp_x, intercept22))
            point15 = [0, intercept22]
            point25 = [elb_hp_x, intercept22]
            x_values5 = [point15[0], point25[0]]
            y_values5 = [point15[1], point25[1]]
            plt.plot(x_values5, y_values5, 'bo', linestyle="--")
            plt.fill_between(x_axis,y2_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y2_axis,intercept22,where = (x_axis>= 0) & (x_axis <= elb_hp_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='Heat pump')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2012'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_hp_x))
            area_ele22 = np.trapz(y= np.array(y2_axis)[idx], x=np.array(x_axis)[idx])
            area_t22 = np.trapz(y2_axis, x_axis , dx =1)
            area_hp22 = area_t22 - area_ele22
            print("The total energy requirement 2012 is",{area_t22})
            print("The total energy delivered by electric is",{area_ele22})
            print("The total energy delivered by heat pump is",{area_hp22})
            plt.grid()
        
        #. HP and electric boiler for 2014
            plt.subplot(2,2,3)
            plt.plot(x_axis, y3_axis,'b--')
            plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
            intercept33 = np.interp(elb_hp_x, x_axis ,y3_axis)
            print(intercept33)
            plt.scatter(elb_hp_x, intercept33, color='black')
            plt.axhline(y=intercept33, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_hp_x, intercept33))
            point16 = [0, intercept33]
            point26 = [elb_hp_x, intercept33]
            x_values6 = [point16[0], point26[0]]
            y_values6 = [point16[1], point26[1]]
            plt.plot(x_values6, y_values6, 'bo', linestyle="--")
            plt.fill_between(x_axis,y3_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y3_axis,intercept33,where = (x_axis>= 0) & (x_axis <= elb_hp_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='Heat pump')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2014'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_hp_x))
            area_ele23 = np.trapz(y= np.array(y3_axis)[idx], x=np.array(x_axis)[idx])
            area_t23 = np.trapz(y3_axis, x_axis , dx =1)
            area_hp23 = area_t23 - area_ele23
            print("The total energy requirement 2014 is",{area_t23})
            print("The total energy delivered by electric is",{area_ele23})
            print("The total energy delivered by heat pump is",{area_hp23})
            plt.grid()
            plt.tight_layout()
        
            plt.show()
        
        #Electric boiler and CHP
        
        #chp and electric boiler 2010
            plt.figure(figsize=(15,10))
            plt.suptitle('Cost using electric boiler and CHP during year of operation', fontsize=14)
            plt.subplot(2,2,1)
            plt.plot(x_axis, y1_axis,'b--')
            plt.axvline(x=elb_chp_x, color='gray', linestyle='--')
            intercept111 = np.interp(elb_chp_x, x_axis ,y1_axis)
            print(intercept111)
            plt.scatter(elb_chp_x, intercept111, color='black')
            plt.axhline(y=intercept111, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_chp_x, intercept111))
            point17 = [0, intercept111]
            point27 = [elb_chp_x, intercept111]
            x_values7 = [point17[0], point27[0]]
            y_values7 = [point17[1], point27[1]]
            plt.plot(x_values7, y_values7, 'bo', linestyle="--")
            plt.fill_between(x_axis,y1_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y1_axis,intercept111,where = (x_axis>= 0) & (x_axis <= elb_chp_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='CHP')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2010'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_chp_x ))
            area_ele31 = np.trapz(y= np.array(y1_axis)[idx], x=np.array(x_axis)[idx])
            area_t31 = np.trapz(y1_axis, x_axis , dx =1)
            area_chp31 = area_t31 - area_ele31
            print("The total energy requirement 2010 is",{area_t31})
            print("The total energy delivered by electric is",{area_ele31})
            print("The total energy delivered by CHP is",{area_chp31})
            plt.grid()
        
        #. cHP and electric boiler for 2012
            plt.subplot(2,2,2)
            plt.plot(x_axis, y2_axis,'b--')
            plt.axvline(x=elb_chp_x, color='gray', linestyle='--')
            intercept222 = np.interp(elb_chp_x, x_axis ,y2_axis)
            print(intercept222)
            plt.scatter(elb_chp_x, intercept222, color='black')
            plt.axhline(y=intercept222, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_chp_x, intercept222))
            point18 = [0, intercept222]
            point28 = [elb_chp_x, intercept222]
            x_values8 = [point18[0], point28[0]]
            y_values8 = [point18[1], point28[1]]
            plt.plot(x_values8, y_values8, 'bo', linestyle="--")
            plt.fill_between(x_axis,y2_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y2_axis,intercept222,where = (x_axis>= 0) & (x_axis <= elb_chp_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='CHP')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2012'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_chp_x ))
            area_ele32 = np.trapz(y= np.array(y2_axis)[idx], x=np.array(x_axis)[idx])
            area_t32 = np.trapz(y2_axis, x_axis , dx =1)
            area_chp32 = area_t32 - area_ele32
            print("The total energy requirement 2012 is",{area_t32})
            print("The total energy delivered by electric is",{area_ele32})
            print("The total energy delivered by CHP is",{area_chp32})
            plt.grid()
        
            #. cHP and electric boiler for 2014
            plt.subplot(2,2,3)
            plt.plot(x_axis, y3_axis,'b--')
            plt.axvline(x=elb_chp_x, color='gray', linestyle='--')
            intercept333 = np.interp(elb_chp_x, x_axis ,y3_axis)
            print(intercept333)
            plt.scatter(elb_chp_x, intercept333, color='black')
            plt.axhline(y=intercept333, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_chp_x, intercept333))
            point19 = [0, intercept333]
            point29 = [elb_chp_x, intercept333]
            x_values9 = [point19[0], point29[0]]
            y_values9 = [point19[1], point29[1]]
            plt.plot(x_values9, y_values9, 'bo', linestyle="--")
            plt.fill_between(x_axis,y3_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis, y3_axis,intercept333,where = (x_axis>= 0) & (x_axis <= elb_chp_x ), color='red')
            plt.legend(['Duration curve for 2014'])
            pop_a = mpatches.Patch(color='green', label='CHP')
            pop_b = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_b,pop_a])
            plt.title(['Duration curve for 2014'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_chp_x ))
            area_ele33 = np.trapz(y= np.array(y3_axis)[idx], x=np.array(x_axis)[idx])
            area_t33 = np.trapz(y3_axis, x_axis , dx =1)
            area_chp33 = area_t33 - area_ele33
            print("The total energy requirement 2014 is",{area_t33})
            print("The total energy delivered by electric is",{area_ele33})
            print("The total energy delivered by CHP is",{area_chp33})
            plt.grid()
            plt.tight_layout()
        
            plt.show()
        
        #Electric boiler,  heat pump and CHP
        
            plt.figure(figsize=(15,10))
            plt.suptitle('Cost using electric boiler heatpump and CHP during year of operation', fontsize=14)
            plt.subplot(2,2,1)
            plt.plot(x_axis, y1_axis,'b--')
            plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
            intercept11 = np.interp(elb_hp_x, x_axis ,y1_axis)
            print(intercept11)
            plt.scatter(elb_hp_x, intercept11, color='black')
            plt.axhline(y=intercept11, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_hp_x, intercept11))
            point1a = [0, intercept11]
            point2a = [elb_hp_x, intercept11]
            x_valuesa = [point1a[0], point2a[0]]
            y_valuesa = [point1a[1], point2a[1]]
            plt.plot(x_valuesa, y_valuesa, 'bo', linestyle="--")
        
            plt.axvline(x=hp_chp_x, color='gray', linestyle='--')
            intercept11a = np.interp(hp_chp_x, x_axis ,y1_axis)
            print(intercept11a)
            plt.scatter(hp_chp_x, intercept11a, color='black')
            plt.axhline(y=intercept11a, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(hp_chp_x, intercept11a))
            point1b = [0, intercept11a]
            point2b = [hp_chp_x, intercept11a]
            x_valuesb = [point1b[0], point2b[0]]
            y_valuesb = [point1b[1], point2b[1]]
            plt.plot(x_valuesb, y_valuesb, 'g', linestyle="--")
            plt.fill_between(x_axis,y1_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis,y1_axis,intercept11a,where = (x_axis>= 0) & (x_axis <= hp_chp_x), color='blue')
            plt.fill_between(x_axis, y1_axis,intercept11,where = (x_axis>= 0) & (x_axis <= elb_hp_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='CHP')
            pop_b = mpatches.Patch(color='blue', label='Heat Pump')
            pop_c = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_c,pop_b, pop_a])
            plt.title(['Duration curve for 2010'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_hp_x ))
            idx1 = np.where((np.array(x_axis)>=elb_hp_x ) & (np.array(x_axis)<=hp_chp_x))
            area_ele41 = np.trapz(y= np.array(y1_axis)[idx], x=np.array(x_axis)[idx])
            area_hp41 = np.trapz(y= np.array(y1_axis)[idx1], x=np.array(x_axis)[idx1])
            area_t41 = np.trapz(y1_axis, x_axis , dx =1)
            area_chp41 = area_t41 - (area_ele41 + area_hp41)
            print("The total energy requirement 2010 is",{area_t41})
            print("The total energy delivered by electric is",{area_ele41})
            print("The total energy delivered by HP is",{area_hp41})
            print("The total energy delivered by CHP is",{area_chp41})
            plt.grid()
        
        #. chp HP and electric boiler for 2012
            plt.subplot(2,2,2)
            plt.plot(x_axis, y2_axis,'b--')
            plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
            intercept22 = np.interp(elb_hp_x, x_axis ,y2_axis)
            print(intercept22)
            plt.scatter(elb_hp_x, intercept22, color='black')
            plt.axhline(y=intercept22, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_hp_x, intercept22))
            point1c = [0, intercept22]
            point2c = [elb_hp_x, intercept22]
            x_valuesc = [point1c[0], point2c[0]]
            y_valuesc = [point1c[1], point2c[1]]
            plt.plot(x_valuesc, y_valuesc, 'bo', linestyle="--")
            plt.axvline(x=hp_chp_x, color='gray', linestyle='--')
            intercept22a = np.interp(hp_chp_x, x_axis ,y2_axis)
            print(intercept22a)
            plt.scatter(hp_chp_x, intercept22a, color='black')
            plt.axhline(y=intercept22a, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(hp_chp_x, intercept22a))
            point1d = [0, intercept22a]
            point2d = [hp_chp_x, intercept22a]
            x_valuesd = [point1d[0], point2d[0]]
            y_valuesd = [point1d[1], point2d[1]]
            plt.plot(x_valuesd, y_valuesd, 'g', linestyle="--")
            plt.fill_between(x_axis,y2_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis,y2_axis,intercept22a,where = (x_axis>= 0) & (x_axis <= hp_chp_x), color='blue')
            plt.fill_between(x_axis, y2_axis,intercept22,where = (x_axis>= 0) & (x_axis <= elb_hp_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='CHP')
            pop_b = mpatches.Patch(color='blue', label='Heat Pump')
            pop_c = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_c,pop_b, pop_a])
            plt.title(['Duration curve for 2012'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_hp_x ))
            idx1 = np.where((np.array(x_axis)>=elb_hp_x ) & (np.array(x_axis)<=hp_chp_x))
            area_ele42 = np.trapz(y= np.array(y2_axis)[idx], x=np.array(x_axis)[idx])
            area_hp42 = np.trapz(y= np.array(y2_axis)[idx1], x=np.array(x_axis)[idx1])
            area_t42 = np.trapz(y2_axis, x_axis , dx =1)
            area_chp42 = area_t42 - (area_ele42 + area_hp42)
            print("The total energy requirement 2012 is",{area_t42})
            print("The total energy delivered by electric is",{area_ele42})
            print("The total energy delivered by HP is",{area_hp42})
            print("The total energy delivered by CHP is",{area_chp42})
            plt.grid()
        
        #. chp HP and electric boiler for 2014
            plt.subplot(2,2,3)
            plt.plot(x_axis, y3_axis,'b--')
            plt.axvline(x=elb_hp_x, color='gray', linestyle='--')
            intercept33 = np.interp(elb_hp_x, x_axis ,y3_axis)
            print(intercept33)
            plt.scatter(elb_hp_x, intercept33, color='black')
            plt.axhline(y=intercept33, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(elb_hp_x, intercept33))
            point1e = [0, intercept33]
            point2e = [elb_hp_x, intercept33]
            x_valuese = [point1e[0], point2e[0]]
            y_valuese = [point1e[1], point2e[1]]
            plt.plot(x_valuese, y_valuese, 'bo', linestyle="--")
            plt.axvline(x=hp_chp_x, color='gray', linestyle='--')
            intercept33a = np.interp(hp_chp_x, x_axis ,y3_axis)
            print(intercept33a)
            plt.scatter(hp_chp_x, intercept33a, color='black')
            plt.axhline(y=intercept33a, color='gray', linestyle='--')
            plt.annotate('point of intersection',xy=(hp_chp_x, intercept33a))
            point1f = [0, intercept33a]
            point2f = [hp_chp_x, intercept33a]
            x_valuesf = [point1f[0], point2f[0]]
            y_valuesf = [point1f[1], point2f[1]]
            plt.plot(x_valuesf, y_valuesf, 'g', linestyle="--")
            plt.fill_between(x_axis,y3_axis,where = (x_axis>= 0) & (x_axis <= 8760), color='green')
            plt.fill_between(x_axis,y3_axis,intercept33a,where = (x_axis>= 0) & (x_axis <= hp_chp_x), color='blue')
            plt.fill_between(x_axis, y3_axis,intercept33,where = (x_axis>= 0) & (x_axis <= elb_hp_x ), color='red')
            pop_a = mpatches.Patch(color='green', label='CHP')
            pop_b = mpatches.Patch(color='blue', label='Heat Pump')
            pop_c = mpatches.Patch(color='red', label='Electric boiler')
            plt.legend(handles=[pop_c,pop_b, pop_a])
            plt.title(['Duration curve for 2014'])
            plt.xlabel("Time(hrs)")
            plt.ylabel("Power(kw)")
            plt.grid()
            idx = np.where((np.array(x_axis)>=0) & (np.array(x_axis)<=elb_hp_x ))
            idx1 = np.where((np.array(x_axis)>=elb_hp_x ) & (np.array(x_axis)<=hp_chp_x))
            area_ele43 = np.trapz(y= np.array(y3_axis)[idx], x=np.array(x_axis)[idx])
            area_hp43 = np.trapz(y= np.array(y3_axis)[idx1], x=np.array(x_axis)[idx1])
            area_t43 = np.trapz(y3_axis, x_axis , dx =1)
            area_chp43 = area_t43 - (area_ele43 + area_hp43)
            print("The total energy requirement 2014 is",{area_t43})
            print("The total energy delivered by electric is",{area_ele43})
            print("The total energy delivered by HP is",{area_hp43})
            print("The total energy delivered by CHP is",{area_chp43})
            plt.tight_layout()
        
            plt.show()
        
        
        
        # Cost Comparison 
            f_chp = 1000
            f_hp = 800
            f_bio = 500
            f_elb = 60
        
        
        
            FC_SYS1A = (f_bio * intercept1  + f_elb * (16383 - intercept1))
            FC_SYS1B = (f_bio * intercept2  + f_elb * (13900 - intercept1)) 
            FC_SYS1C = (f_bio * intercept3  + f_elb * (14000 - intercept1)) 
            FC_SYS2A = (f_elb * (16383 - intercept11)  + f_hp * intercept11) 
            FC_SYS2B = (f_elb * (13900 - intercept22)  + f_hp * intercept22) 
            FC_SYS2C = (f_elb * (14000 - intercept33) + f_hp * intercept33) 
            FC_SYS3A = (f_elb * (16383 - intercept111) + f_chp * intercept111) 
            FC_SYS3B = (f_elb * (13900 - intercept222) + f_chp * intercept222) 
            FC_SYS3C = (f_elb * (14000 - intercept333) + f_chp * intercept333) 
            FC_SYS4A = (f_elb * (16383 - intercept11) + f_hp * (intercept11 - intercept11a) + f_chp * intercept11a) 
            FC_SYS4B = (f_elb * (13900 - intercept22)+ f_hp * (intercept22 - intercept22a) + f_chp * intercept22a) 
            FC_SYS4C = (f_elb * (14000 - intercept33) + f_hp * (intercept33 - intercept33a) + f_chp * intercept33a) 
        
            v_chp = en_chp * af_chp * (1+ pf_chp)/eff_chp
            v_hp = (en_hp/COP)
            v_bio = (en_bio/eff_bio)
            v_elb =(en_elb/eff_elb)
        
            en_chp = 0.01
            en_hp = 0.12
            en_bio = 0.01
            en_elb = 0.3
        
            EC_SYS1A = (v_bio * area_bio11 + v_elb * area_ele11 ) 
            EC_SYS1B = (v_bio * area_bio12 + v_elb * area_ele12 ) 
            EC_SYS1C = (v_bio * area_bio13 + v_elb * area_ele13 ) 
            EC_SYS2A = (v_elb * area_ele21  + v_hp * area_hp21) 
            EC_SYS2B = (v_elb * area_ele22  + v_hp * area_hp22) 
            EC_SYS2C = (v_elb * area_ele23  + v_hp * area_hp23) 
            EC_SYS3A = (v_elb * area_ele31 + v_chp * area_chp31) 
            EC_SYS3B = (v_elb * area_ele32 + v_chp * area_chp32) 
            EC_SYS3C = (v_elb * area_ele33 + v_chp * area_chp33) 
            EC_SYS4A = (v_elb * area_ele41 + v_hp * area_hp41 + v_chp * area_chp41) 
            EC_SYS4B = (v_elb * area_ele42 + v_hp * area_hp42 + v_chp * area_chp42) 
            EC_SYS4C = (v_elb * area_ele43 + v_hp * area_hp43 + v_chp * area_chp43) 
        
        
            TC_SYS1A = FC_SYS1A + EC_SYS1A
            TC_SYS1B = FC_SYS1B + EC_SYS1B
            TC_SYS1C = FC_SYS1C + EC_SYS1C
            TC_SYS2A = FC_SYS2A + EC_SYS2A
            TC_SYS2B = FC_SYS2B + EC_SYS2B
            TC_SYS2C = FC_SYS2C + EC_SYS2C
            TC_SYS3A = FC_SYS3A + EC_SYS3A
            TC_SYS3B = FC_SYS3B + EC_SYS3B
            TC_SYS3C = FC_SYS3C + EC_SYS3C
            TC_SYS4A = FC_SYS4A + EC_SYS4A
            TC_SYS4B = FC_SYS4B + EC_SYS4B
            TC_SYS4C = FC_SYS4C + EC_SYS4C
        
            FCAs = [FC_SYS1A, FC_SYS2A, FC_SYS3A, FC_SYS4A]
            FCBs = [FC_SYS1B, FC_SYS2B, FC_SYS3B, FC_SYS4B]
            FCCs = [FC_SYS1C, FC_SYS2C, FC_SYS3C, FC_SYS4C]
        
            VCAs = [EC_SYS1A, EC_SYS2A, EC_SYS3A, EC_SYS4A]
            VCBs = [EC_SYS1B, EC_SYS2B, EC_SYS3B, EC_SYS4B]
            VCCs = [EC_SYS1C, EC_SYS2C, EC_SYS3C, EC_SYS4C]
        
            TCAs = [TC_SYS1A, TC_SYS2A, TC_SYS3A, TC_SYS4A]
            TCBs = [TC_SYS1B, TC_SYS2B, TC_SYS3B, TC_SYS4B]
            TCCs = [TC_SYS1C, TC_SYS2C, TC_SYS3C, TC_SYS4C]
        
            SYS = ["ELEC & BIO", "ELEC & HP", "ELE & CHP", "ELEC , HP & CHP"]
        
            plt.figure(figsize=(10,8))
            plt.scatter(SYS, FCAs, color='red')
            plt.scatter(SYS, FCBs, color='green')
            plt.scatter(SYS, FCCs, color='blue')
            plt.scatter(SYS, VCAs, color='red')
            plt.scatter(SYS, VCBs, color='green')
            plt.scatter(SYS, VCCs, color='blue')
            plt.scatter(SYS, TCAs, color='red')
            plt.scatter(SYS, TCBs, color='green')
            plt.scatter(SYS, TCCs, color='blue')
        
            plt.plot(SYS, FCAs, color='red')
            plt.plot(SYS, FCBs, color='green')
            plt.plot(SYS, FCCs, color='blue')
            plt.text(0, TC_SYS1A, 'Total cost', color='red',
                 rotation=40, rotation_mode='anchor')
            plt.text(0, TC_SYS1B, 'Total cost', color='green',
                 rotation=30, rotation_mode='anchor')
            plt.text(0, TC_SYS1C, 'Total cost', color='blue',
                 rotation=20, rotation_mode='anchor')
            plt.text(1.5, EC_SYS2A, 'Operational cost', color='red',
                 rotation=0, rotation_mode='anchor')
            plt.text(1.5, EC_SYS2B, 'Operational cost', color='green',
                 rotation=0, rotation_mode='anchor')
            plt.text(1.5, EC_SYS2C, 'operational cost', color='blue',
                 rotation=0, rotation_mode='anchor')
            plt.text(2.5, FC_SYS3A, 'Fixed Cost', color='red',
                 rotation=0, rotation_mode='anchor')
            plt.text(2.5, FC_SYS3B, 'Fixed cost', color='green',
                 rotation=-2, rotation_mode='anchor')
            plt.text(2.5, FC_SYS3C, 'Fixed cost', color='blue',
                 rotation=0, rotation_mode='anchor')
            plt.legend(['2010', '2012','2014'])
            plt.title(['Cost for different energy system combination at different year'])
            plt.xlabel("SYSTEM COMBINATION")
            plt.ylabel("COST (EURO)")
        
            plt.plot(SYS, VCAs, color='red', linestyle='--')
            plt.plot(SYS, VCBs, color='green', linestyle='--')
            plt.plot(SYS, VCCs, color='blue', linestyle='--')
        
            plt.plot(SYS, TCAs, color='red', linestyle='-.')
            plt.plot(SYS, TCBs, color='green', linestyle='-.')
            plt.plot(SYS, TCCs, color='blue', linestyle='-.')
        
            plt.legend(['2010', '2012','2014'])
        #plt.title(['Operational cost for different energy system combination at different year'])
            plt.xlabel("SYSTEM COMBINATION")
            plt.ylabel("COST (EURO)")
            plt.grid()
            plt.show()
        
        
        #Energy Scenarios
        
        #1. Peak load 
        
        
            First_year = [area_ele11, area_ele21, area_ele31, area_ele41]          
            Second_year =  [area_ele12, area_ele22, area_ele32, area_ele42]  
            Third_year = [area_ele13, area_ele23, area_ele33, area_ele43] 
            systems = ['ELE & BIO', 'ELE & HP', 'ELE & CHP', 'ELE, HP & CHP']
        
        
        # Creating explode data
            explode = (0.1, 0.0, 0.2, 0.3)
         
        # Creating color parameters
            colors = ( "orange", "cyan", "brown",
                  "grey", "indigo", "blue")
         
        # Wedge properties
            wp = { 'linewidth' : 1, 'edgecolor' : "green" }
         
        # Creating autocpt arguments
            def func(pct, allvalues):
                absolute = int(pct / 100.*np.sum(allvalues))
                return "{:.1f}%\n({:d} KWH)".format(pct, absolute)
         
        # Creating plot
            fig, ax = plt.subplots(figsize =(10, 7))
            wedges, texts, autotexts = ax.pie(First_year,
                                              autopct = lambda pct: func(pct, First_year),
                                              explode = explode,
                                              labels = systems,
                                              shadow = True,
                                              colors = colors,
                                              startangle = 90,
                                              wedgeprops = wp,
                                              textprops = dict(color ="magenta"))
         
        # Adding legend
            ax.legend(wedges, First_year,
                      title ="Systems",
                      loc ="center left",
                      bbox_to_anchor =(1, 0, 0.5, 1))
         
            plt.setp(autotexts, size = 8, weight ="bold")
            ax.set_title("Peak load scenarios for 2010")
         
        
        # Creating explode data
            explode = (0.1, 0.0, 0.2, 0.3)
         
        # Creating color parameters
            colors = ( "orange", "black", "brown",
                  "grey", "indigo", "beige")
            
        # Wedge properties
            wp = { 'linewidth' : 1, 'edgecolor' : "green" }
         
        # Creating autocpt arguments
            def func1(pct, allvalues):
                absolute = int(pct / 100.*np.sum(allvalues))
                return "{:.1f}%\n({:d} KWH)".format(pct, absolute)
         
        # Creating plot
            fig, ax = plt.subplots(figsize =(10, 7))
            wedges, texts, autotexts = ax.pie(Second_year,
                                              autopct = lambda pct: func1(pct, Second_year),
                                              explode = explode,
                                              labels = systems,
                                              shadow = True,
                                              colors = colors,
                                              startangle = 90,
                                              wedgeprops = wp,
                                              textprops = dict(color ="magenta"))
         
        # Adding legend
            ax.legend(wedges, Second_year,
                      title ="Systems",
                      loc ="center left",
                      bbox_to_anchor =(1, 0, 0.5, 1))
         
            plt.setp(autotexts, size = 8, weight ="bold")
            ax.set_title("Peak load scenarios for 2012")
        
        
        
        # Creating explode data
            explode = (0.1, 0.0, 0.2, 0.3)
         
        # Creating color parameters
            colors = ( "red", "cyan", "brown",
                  "grey", "indigo", "beige")
         
        # Wedge properties
            wp = { 'linewidth' : 1, 'edgecolor' : "green" }
         
        # Creating autocpt arguments
            def func2(pct, allvalues):
                absolute = int(pct / 100.*np.sum(allvalues))
                return "{:.1f}%\n({:d} KWH)".format(pct, absolute)
         
        # Creating plot
            fig, ax = plt.subplots(figsize =(10, 7))
            wedges, texts, autotexts = ax.pie(Third_year,
                                              autopct = lambda pct: func2(pct, Third_year),
                                              explode = explode,
                                              labels = systems,
                                              shadow = True,
                                              colors = colors,
                                              startangle = 90,
                                              wedgeprops = wp,
                                              textprops = dict(color ="magenta"))
         
        # Adding legend
            ax.legend(wedges, Third_year,
                      title ="Systems",
                      loc ="center left",
                      bbox_to_anchor =(1, 0, 0.5, 1))
         
            plt.setp(autotexts, size = 8, weight ="bold")
            ax.set_title("Peak load scenarios for 2014")
        
            plt.show()
        
        #Total Efficiency 
            
        
        
        
            EFF1 = eff_elb * eff_bio
            EFF2 = eff_elb * eff_car
            EFF3 = eff_elb * eff_chp
            EFF4 = eff_elb * eff_car * eff_chp
        
            S = ['ELB','ELB & BIO','BIO' ,'ELB & HP','CHP', 'ELB & CHP', 'HP','ELB, HP & CHP']
            Z = [eff_elb, EFF1,eff_bio, EFF2, eff_chp, EFF3,eff_car, EFF4]
            plt.xlabel('Systems')
            plt.ylabel('System Efficiency')
            plt.title('Efficiency of the system')
            plt.plot(S, Z, color='red', linestyle='--')
            plt.legend(['Efficiency'])
            plt.grid()
        
            plt.show()
            
        
            #first year 
            ele_bio_tc_1.append(TC_SYS1A)
            ele_hp_tc_1.append(TC_SYS2A)
            ele_chp_tc_1.append(TC_SYS3A)
            ele_hp_chp_tc_1.append(TC_SYS4A)
        
            ele_bio_fc_1.append(FC_SYS1A)
            ele_hp_fc_1.append(FC_SYS2A)
            ele_chp_fc_1.append(FC_SYS3A)
            ele_hp_chp_fc_1.append(FC_SYS4A)
        
            ele_bio_oc_1.append(EC_SYS1A)
            ele_hp_oc_1.append(EC_SYS2A)
            ele_chp_oc_1.append(EC_SYS3A)
            ele_hp_chp_oc_1.append(EC_SYS4A)
        
            #Second year
            ele_bio_tc_2.append(TC_SYS1B)
            ele_hp_tc_2.append(TC_SYS2B)
            ele_chp_tc_2.append(TC_SYS3B)
            ele_hp_chp_tc_2.append(TC_SYS4B)
        
            ele_bio_fc_2.append(FC_SYS1B)
            ele_hp_fc_2.append(FC_SYS2B)
            ele_chp_fc_2.append(FC_SYS3B)
            ele_hp_chp_fc_2.append(FC_SYS4B)
        
            ele_bio_oc_2.append(EC_SYS1B)
            ele_hp_oc_2.append(EC_SYS2B)
            ele_chp_oc_2.append(EC_SYS3B)
            ele_hp_chp_oc_2.append(EC_SYS4B)
        
            #Third year
        
            ele_bio_tc_3.append(TC_SYS1C)
            ele_hp_tc_3.append(TC_SYS2C)
            ele_chp_tc_3.append(TC_SYS3C)
            ele_hp_chp_tc_3.append(TC_SYS4C)
        
            ele_bio_fc_3.append(FC_SYS1C)
            ele_hp_fc_3.append(FC_SYS2C)
            ele_chp_fc_3.append(FC_SYS3C)
            ele_hp_chp_fc_3.append(FC_SYS4C)
        
            ele_bio_oc_3.append(EC_SYS1C)
            ele_hp_oc_3.append(EC_SYS2C)
            ele_chp_oc_3.append(EC_SYS3C)
            ele_hp_chp_oc_3.append(EC_SYS4C)
            
            print('END LOOP')
        
        TTCA = [ele_bio_tc_1, ele_hp_tc_1, ele_chp_tc_1, ele_hp_chp_tc_1]
        
        FFCA = [ele_bio_fc_1, ele_hp_fc_1, ele_chp_fc_1, ele_hp_chp_fc_1]
        
        OOCA = [ele_bio_oc_1, ele_hp_oc_1, ele_chp_oc_1, ele_hp_chp_oc_1]
        
        TTCB = [ele_bio_tc_2, ele_hp_tc_2, ele_chp_tc_2, ele_hp_chp_tc_2]
        
        FFCB = [ele_bio_fc_2, ele_hp_fc_2, ele_chp_fc_2, ele_hp_chp_fc_2]
        
        OOCB = [ele_bio_oc_2, ele_hp_oc_2, ele_chp_oc_2, ele_hp_chp_oc_2]
            
        TTCC = [ele_bio_tc_3, ele_hp_tc_3, ele_chp_tc_3, ele_hp_chp_tc_3]
        
        FFCC = [ele_bio_fc_3, ele_hp_fc_3, ele_chp_fc_3, ele_hp_chp_fc_3]
        
        OOCC = [ele_bio_oc_3, ele_hp_oc_3, ele_chp_oc_3, ele_hp_chp_oc_3]
        
        elb_bio_tcc = [ele_bio_tc_1, ele_bio_tc_2, ele_bio_tc_3] 
        ele_bio_occ = [ele_bio_oc_1, ele_bio_oc_2, ele_bio_oc_3]
        ele_bio_fcc = [ele_bio_fc_1, ele_bio_fc_2, ele_bio_fc_3]
        
        
        plt.figure(figsize=(15,10))
        Y = ['1st', '2nd', '3rd', '4th', '5th']
        plt.scatter(Y, ele_bio_tc_1, c ="pink",
                    linewidths = 2,
                    marker ="s",
                    edgecolor ="green",
                    s = 50)
        plt.scatter(Y, ele_bio_tc_2, c ="yellow",
                    linewidths = 2,
                    marker ="^",
                    edgecolor ="red",
                    s = 200)
        plt.scatter(Y, ele_bio_tc_3,c ="blue",
                    linewidths = 2,
                    marker ="^",
                    edgecolor ="brown",
                    s = 200)
        
        plt.scatter(Y, ele_bio_fc_1,c ="pink",
                    linewidths = 2,
                    marker ="s",
                    edgecolor ="orange",
                    s = 50)
        plt.scatter(Y, ele_bio_fc_2,linewidths = 2,
        marker ="^",
        edgecolor ="red",
        s = 200)
        plt.scatter(Y, ele_bio_fc_3, c ="blue",
                    linewidths = 2,
                    marker ="^",
                    edgecolor ="pink",
                    s = 200)
        
        plt.scatter(Y, ele_bio_oc_1, c ="pink",
                    linewidths = 2,
                    marker ="s",
                    edgecolor ="black",
                    s = 50)
        plt.scatter(Y, ele_bio_oc_2, c ="yellow",
                    linewidths = 2,
                    marker ="^",
                    edgecolor ="gray",
                    s = 200)
        plt.scatter(Y, ele_bio_oc_3, c ="blue",
                    linewidths = 2,
                    marker ="^",
                    edgecolor ="black",
                    s = 200)
        plt.xlabel('10% increase in fixed investment cost in each steps')
        plt.ylabel('Cost (EURO)')
        plt.title('System Combination of electric and bio boiler')
        plt.grid()
        plt.legend(["Total cost 2010" , "Total cost 2012", "Total cost 2014", "Investment cost 2010", "Investment cost 2012", "Investment cost 2014", "Operational cost 2010","Operational cost 2012", "Operational cost 2014"])
        plt.show()
