import tkinter as tk
from tkinter import ttk
from PMPrediction import *

root = tk.Tk()
root.title("PM Prediction")

# Add a grid
mainframe = tk.Frame(root)
mainframe.grid(column=0, row=0)
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.pack(pady=100, padx=100)

def var_states():
    #Status Message placeholders
    statusMsg = tk.StringVar()
    statusMsg.set("")
    tk.Label(mainframe, textvariable=statusMsg, anchor="w").grid(row=20, column=2, columnspan=3, sticky="W", padx=10,
                                                                 pady=10)
    processStatusMsg = tk.StringVar()
    processStatusMsg.set("")
    tk.Label(mainframe, textvariable=processStatusMsg, anchor="w").grid(row=21, column=2, columnspan=3, sticky="W",
                                                                        padx=10, pady=10)

    if (modelChoice.get() == 'StaticAndPersonal'):
        modelInstance = StaticAndPersonal(modelChoice.get(), tkDsChoice.get(), tkvar1.get(), tkvar2.get(), tkvar3.get())
    else:
        modelInstance = StaticOnly(modelChoice.get(), tkDsChoice.get(), tkvar1.get(), tkvar2.get(), tkvar3.get())

    # modelInstance.preprocess(statusMsg, root)
    statusMsg.set("Preprocessing completed.")
    processStatusMsg.set("Training and prediction in progress")
    root.update()

    # trueValues, predictedValues = modelInstance.predictSpatial(statusMsg, root)
    # maeSpatial, mapeSpatial = modelInstance.evaluate(trueValues, predictedValues)
    # statusMsg.set("Preprocessing, Spatial Prediction completed. Temporal prediction in progress")
    # processStatusMsg.set("Spatial Prediction:: MAE::" + str(maeSpatial) + "   MAPE::" + str(mapeSpatial))
    # root.update()
    #
    # trueValues, predictedValues = modelInstance.predictTemporal(statusMsg, root)
    # maeTemporal, mapeTemporal = modelInstance.evaluate(trueValues, predictedValues)
    # statusMsg.set("Preprocessing, Spatial, Temporal Prediction completed. Spatio-Temporal prediction in progress")
    # processStatusMsg.set("Spatial Prediction:: MAE::" + str(maeSpatial) + "   MAPE::" + str(mapeSpatial)
    #                      + "   Temporal Prediction:: MAE::" + str(maeTemporal) + "   MAPE::" + str(mapeTemporal))
    # root.update()
    #
    avg_mape_van, avg_mape_reg, avg_mape_bid, avg_mape_gru = modelInstance.predictSpatioTemporal()
    statusMsg.set("Spatio Temporal Prediction Completed")
    processStatusMsg.set("Average MAPEs :: Van: " + str(avg_mape_van) + ", Reg: " + str(avg_mape_reg) +
                         ", Bid: " + str(avg_mape_bid) + ", GRU: " + str(avg_mape_gru))
    root.update()
    # maeSTemporal, mapeSTemporal = modelInstance.evaluate(trueValues, predictedValues)
    # statusMsg.set("Spatio Temporal Prediction Completed")
    # # processStatusMsg.set("Spatial Prediction:: MAE::{0}   MAPE::{1}   Temporal Prediction:: MAE::{2}   MAPE::{3}   Spatio Temporal Prediction:: MAE::{4}   MAPE::{5}".format(
    # #     str(maeSpatial), str(mapeSpatial), str(maeTemporal), str(mapeTemporal), str(maeSTemporal), str(mapeSTemporal)))
    # processStatusMsg.set("Spatio Temporal Prediction:: MAE::" + str(maeSTemporal) + "   MAPE::" + str(mapeSTemporal))
    # root.update()

# Create a Tkinter variable
tkDsChoice = tk.StringVar()
tkDsChoice.set('INHALE')  # set the default option
popupMenu = ttk.Combobox(mainframe, textvariable=tkDsChoice, state="readonly")
popupMenu['values'] = ['INHALE', 'DAPHNE', 'London', 'PEEPS']
popupMenu.current(0)
tk.Label(mainframe, text="DataSet", anchor="w").grid(row=1, column=1, sticky="W", padx=20, pady=10)
popupMenu.grid(row=1, column=2, padx=10, pady=10, sticky="W")

tk.Label(mainframe, text="Model", anchor="w").grid(row=3, column=1, sticky="W", padx=20, pady=10)
modelChoice = tk.StringVar()
modelChoice.set("StaticAndPersonal")
tk.Radiobutton(mainframe,
               text="Static Sensor only",
               # padx = 20,
               variable=modelChoice,
               value='Static').grid(row=3, column=2, padx=10, pady=10, sticky="W")
tk.Radiobutton(mainframe,
               text="Static and Personal",
               # padx = 20,
               variable=modelChoice,
               value='StaticAndPersonal').grid(row=3, column=3, padx=10, pady=10, sticky="W")


# Create a Tkinter variable
tkvar1 = tk.StringVar()
tkvar1.set('Extra Trees')  # set the default option

popupMenu1 = ttk.Combobox(mainframe,textvariable= tkvar1, state="readonly")
popupMenu1['values'] = ['Extra Trees', 'Random Forests','XGBoost','Linear Regression']
popupMenu1.current(0)
tk.Label(mainframe, text="Spatial Models", anchor="w").grid(row=13, column=1, sticky="W", padx=20, pady=10)
popupMenu1.grid(row=13, column=2, padx=10, pady=10, sticky="W")

# Dictionary with options
tkvar2 = tk.StringVar()
tkvar2.set('LSTM')  # set the default option

popupMenu2 = ttk.Combobox(mainframe,textvariable= tkvar2, state="readonly")
popupMenu2['values'] =['LSTM', 'GRU']
popupMenu2.current(0)
tk.Label(mainframe, text="Temporal Models", anchor="w").grid(row=15, column=1, sticky="W", padx=20, pady=10)
popupMenu2.grid(row=15, column=2, padx=10, pady=10, sticky="W")

tkvar3 = tk.StringVar()
tkvar3.set('Extra Trees with LSTM')
popupMenu3 = ttk.Combobox(mainframe, textvariable= tkvar3, state="readonly")
popupMenu3['values'] = ['AutoRegressive LSTM with Traffic and LUR', 'Extra Trees with LSTM']
popupMenu3.current(1)
tk.Label(mainframe, text="Spatio Temporal Models", anchor="w").grid(row=17, column=1, sticky="W", padx=20, pady=10)
popupMenu3.grid(row=17, column=2, padx=10, pady=10, sticky="W")

tk.Button(mainframe, text='Predict', command=var_states).grid(row=20, column=1, padx=20, pady=20, sticky="W")


# on change dropdown value
def change_dropdown(*args):
    print(tkDsChoice.get())
    staticList = ['LUR']
    staticPersonalList = ['Extra Trees', 'Random Forests','XGBoost','Linear Regression']
    staticTemporalList = ['LSTM-Entire Sequence','LSTM-Single Output','LSTM-Autoregressive',
        'LSTM-Single Output with Traffic','LSTM-Single Output with Weather', 'LSTM-Single Output with Weather and Traffic'
                        , 'LSTM-Autoregressive with Traffic']
    staticPersonalTemporalList = ['LSTM', 'GRU']
    staticSTList = ['AutoRegressive LSTM with Traffic and LUR']
    staticPersonalSTList = ['Extra Trees with LSTM']
    if modelChoice.get() == 'Static':
        popupMenu1['values'] = staticList
        popupMenu2['values'] = staticTemporalList
        popupMenu3['values'] = staticSTList
    else:
        popupMenu1['values'] = staticPersonalList
        popupMenu2['values'] = staticPersonalTemporalList
        popupMenu3['values'] = staticPersonalSTList
    popupMenu1.current(0)
    popupMenu2.current(0)
    popupMenu3.current(0)
    root.update()


# link function to change dropdown
modelChoice.trace('w', change_dropdown)

root.mainloop()
