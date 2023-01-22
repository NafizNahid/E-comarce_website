from django.shortcuts import render
import numpy as np
from joblib import load
model = load('./savedModels/model.joblib')

# Create your views here.
def mlapp(request):
    if request.method == 'POST':
        battery_power = request.POST['battery_power']
        dual_sim = 'Yes'
        int_memory = request.POST['int_memory']
        mobile_wt = request.POST['mobile_wt']
        ram = request.POST['ram']
        ramg = int(ram)/1000
        talk_time = 20
        touch_screen = 'Yes'
        wifi = 'Yes'
        y = [[battery_power, 1,1, int_memory,mobile_wt,ram,20,1,1]]
        b = np.array(y, dtype=float)
        y_pred = model.predict(b)
        
        n = y_pred[0]*3500
        y_pred =int(n) 
        ramg = int(ramg)  
            
        return render(request, 'mlapp/result.html', {'result' : y_pred , 'battery_power': battery_power,'dual_sim':dual_sim, 'int_memory':int_memory,
                                                     'mobile_wt' :mobile_wt, 'ram':ram ,'talk_time':talk_time,'touch_screen':touch_screen,'wifi':wifi,'ramg':ramg})

    return render(request,'mlapp/ml.html')
