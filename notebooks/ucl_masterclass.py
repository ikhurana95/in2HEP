import numpy as np
import cv2
from PIL import Image
import time
from resizeimage import resizeimage
import imutils
import pickle
from copy import deepcopy
import math
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom


def save_img(img,filePath = "number.pkl"):
    """
    Writes events to pickle file. Ideally dump few objects where the objects could be any data structures
    containing other objects
    :param events:
    :param filePath:
    """
    print("Writing...")
    if filePath[-3:]!="pkl":
        filePath = filePath+".pkl"

    with open(filePath, "wb") as output:

        pickle.dump(img, output, pickle.HIGHEST_PROTOCOL)

def load(filePath,python2 = False):
    """
    Loads objects from pickle file
    :param filePath:
    :return: values in pickle file
    """
    load = []
    print("Loading...")
    with open(filePath, "rb") as file:
        hasNext = True
        if python2:

            load.append(pickle.load(file))
        else:
            load.append(pickle.load(file, encoding='latin1'))
        while hasNext:
            try:
                if python2:
                    load.append(pickle.load(file))
                else:
                    load.append(pickle.load(file, encoding='latin1'))
            except:
                hasNext = False

    if len(load) == 1:
        return load[0]
    else:
        return load


def get_webcam_img():

    """
    Creates an webcam image for the Handwritten Digit Classifier Project.

    Uses cv2 python library to apply filters on captured image so that it can
    be processed by the neural network.

    Saves a 28*28 black and white image.


    """
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Applies black and white filter
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Applies edge detection filter
        edged = cv2.Canny(gray,75,180)
        fgmask = fgbg.apply(edged)

        # Crops the recorded image
        w = 350
        h = 350
        x = int(gray.shape[1]/2 - w/2)
        y = int(gray.shape[0]/2 - h/2)

        crop_img = gray[y:y+h, x:x+w]
        # Image displayed without edge detection
        cv2.imshow("Press SPACE to shoot", crop_img)

        # Uncomment line below to demonstrate edge detection
        # cv2.imshow("Press SPACE to shoot", fgmask)



        # Checks if space is pressed
        if cv2.waitKey(1) & 0xFF == ord(' ') or 0xFF == ord('\n'):

            crop_img = fgmask[y:y+h, x:x+w]

            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(crop_img,kernel,iterations = 1)
            print(dilation.shape)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray  = cv2.resize(dilation,None,fx=128/350, fy=128/350, interpolation = cv2.INTER_CUBIC)
            print(gray.shape)

            gray = cv2.dilate(gray,kernel,iterations = 1)
            gray  = cv2.resize(gray,None,fx=28/128, fy=28/128, interpolation = cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            print(gray.shape)

            cv2.imwrite('digit.png',gray)

            # Saves image to a pickle file
            save_img(gray)
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)

    return gray



##########################
#ATLAS Analysis Functions#
##########################



class_names_grouped = ['VH -> Vbb','Diboson','ttbar','Single top', 'W+(bb,bc,cc,bl)','W+cl','W+ll','Z+(bb,bc,cc,bl)',
                       'Z+cl','Z+ll'
                       ]

class_names_map = {'VH -> Vbb':['ggZllH125','ggZvvH125','qqWlvH125', 'qqZllH125', 'qqZvvH125'],
    'Diboson':['WW','ZZ','WZ'],
    'ttbar':['ttbar'],
    'Single top':['stopWt','stops','stopt'],
    'W+(bb,bc,cc,bl)':['Wbb','Wbc','Wcc','Wbl'],
    'W+cl':['Wcl'],
    'W+ll':['Wl'],
    'Z+(bb,bc,cc,bl)':['Zbb','Zbc','Zcc','Zbl'],
    'Z+cl':['Zcl'],
    'Z+ll':['Zl']
}

colour_map = {'VH -> Vbb':'#FF0000',
    'Diboson':'#999999',
    'ttbar':'#FFCC00',
    'Single top':'#CC9900',
    'W+(bb,bc,cc,bl)':'#006600',
    'W+cl':'#66CC66',
    'W+ll':'#99FF99',
    'Z+(bb,bc,cc,bl)':'#0066CC',
    'Z+cl':'#6699CC',
    'Z+ll':'#99CCFF'
}

legend_names = [r'VH $\rightarrow$ Vbb','Diboson',r"t$\bar t$",'Single top', 'W+(bb,bc,cc,bl)','W+cl','W+ll','Z+(bb,bc,cc,bl)',
                'Z+cl','Z+ll'
                ]


def setBinCategory(df,bins):

    if len(bins)!=21:
        print ("ONLY SET FOR 20 BINS")

    df['bin_scaled'] = 999
    bin_scaled_list = df['bin_scaled'].tolist()

    step = 2/(len(bins)-1)  #step between midpoints
    midpoint = -1 + step/2.0   #Initial midpoint
    decision_value_list = df['decision_value'].tolist()

    for j in range(len(bins)-1):
        for i in range(len(decision_value_list)):
            if ((decision_value_list[i] >= bins[j]) & (decision_value_list[i] < bins[j+1])):
                bin_scaled_list[i] = midpoint
        midpoint = midpoint + step

    df['bin_scaled'] = bin_scaled_list

    return df

def bdt_plot(df,z_s = 10,z_b = 10,show=False, block=False, trafoD_bins = False, bin_number = 20):
    """Plots histogram decision score output of classifier"""

    nJets = df['nJ'].tolist()[1]

    if trafoD_bins == True:
        bins, arg2, arg3 = trafoD_with_error(df)
        print(len(bins))
    else:
         bins = np.linspace(-1,1,bin_number+1)

    # Initialise plot stuff
    plt.ion()
    plt.close("all")
    fig = plt.figure(figsize=(8.5,7))
    plot_range = (-1, 1)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"

    # df = setBinCategory(df,bins)

    decision_value_list = df['bin_scaled'].tolist()
    post_fit_weight_list = df['post_fit_weight'].tolist()
    sample_list = df['sample'].tolist()

    # Get list of hists.
    for t in class_names_grouped[::-1]:
        class_names = class_names_map[t]
        class_decision_vals = []
        plot_weight_vals = []
        for c in class_names:
            for x in range(0,len(decision_value_list)):
                if sample_list[x] == c:
                    class_decision_vals.append(decision_value_list[x])
                    plot_weight_vals.append(post_fit_weight_list[x])

        plot_data.append(class_decision_vals)
        plot_weights.append(plot_weight_vals)
        plot_colors.append(colour_map[t])

    # Plot.
    if nJets == 2:

        multiplier = 20
    elif nJets == 3:
        multiplier = 100

    plt.plot([],[],color='#FF0000',label=r'VH $\rightarrow$ Vbb x '+str(multiplier))

    plt.hist(plot_data,
             bins=bins,
             weights=plot_weights,
             range=plot_range,
             rwidth=1,
             color=plot_colors,
             label=legend_names[::-1],
             stacked=True,
             edgecolor='none')

    df_sig = df.loc[df['Class']==1]



    plt.hist(df_sig['bin_scaled'].tolist(),
         bins=bins,
         weights=(df_sig['post_fit_weight']*multiplier).tolist(),
         range=plot_range,
         rwidth=1,
         histtype = 'step',
         linewidth=2,
         color='#FF0000',
         edgecolor='#FF0000')

    x1, x2, y1, y2 = plt.axis()
    plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([5,135000])
    axes.set_xlim([-1,1])
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    plt.xticks(x, x,fontweight = 'normal',fontsize = 20)
    y = [r"10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
    plt.yticks(yi, y,fontweight = 'normal',fontsize = 20)

    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='major', direction='in', length=10, width=1)
    axes.yaxis.set_tick_params(which='minor', direction='in', length=5, width=1)

    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='major', direction='in', length=10, width=1)
    axes.xaxis.set_tick_params(which='minor', direction='in', length=5, width=1)

    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()

    #Weird hack thing to get legend entries in correct order
    handles = handles[::-1]
    handles = handles+handles
    handles = handles[1:12]

    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles)

    plt.ylabel("Events",fontsize = 20,fontweight='normal')
    axes.yaxis.set_label_coords(-0.07,0.93)
    plt.xlabel(r"BDT$_{VH}$ output",fontsize = 20,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS Internal", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)

    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)

    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    plt.show(block=block)


    return fig,axes



def plot_variable(df,variable, bins = None,bin_number = 20):
    """
    Takes a pandas df and plots a specific variable (mBB, Mtop etc)

    """

    nJets = 2

#     if bins == None:
#         bins = np.linspace(0,400,bin_number+1)
#     print(bins)
    # Initialise plot stuff
    bins = 20
    plt.ion()
    plt.close("all")
    fig = plt.figure(figsize=(8.5*1.2,7*1.2))
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"


    var_list = df[variable].tolist()

    if variable in ['mBB','Mtop','pTV','MET','mTW','pTB1','pTB2']:
        var_list = [i/1e3 for i in var_list]


    post_fit_weight_list = df['post_fit_weight'].tolist()
    sample_list = df['sample'].tolist()

    # Get list of hists.
    for t in class_names_grouped[::-1]:
        class_names = class_names_map[t]
        class_decision_vals = []
        plot_weight_vals = []
        for c in class_names:
            for x in range(0,len(var_list)):
                if sample_list[x] == c:
                    class_decision_vals.append(var_list[x])
                    plot_weight_vals.append(post_fit_weight_list[x])

        plot_data.append(class_decision_vals)
        plot_weights.append(plot_weight_vals)
        plot_colors.append(colour_map[t])

    multiplier = 20


    data = plt.hist(plot_data,
             bins=bins,
             weights=plot_weights,
             rwidth=1,
             color=plot_colors,
             label=legend_names[::-1],
             stacked=True,
             edgecolor='none')

    df_sig = df.loc[df['Class']==1]
    var_list_sig = df_sig[variable].tolist()


    if variable in ['mBB','Mtop','pTV','MET','mTW']:
        var_list_sig = [i/1e3 for i in var_list_sig]
    plt.hist(var_list_sig,
         bins=bins,
         weights=(df_sig['post_fit_weight']*multiplier).tolist(),
         rwidth=1,
         histtype = 'step',
         linewidth=2,
         color='#FF0000',
         edgecolor='#FF0000')
    plt.plot([],[],color='#FF0000',label=r'VH $\rightarrow$ Vbb x '+str(multiplier))

    x1, x2, y1, y2 = plt.axis()
    axes = plt.gca()
    plt.xticks(fontweight = 'normal',fontsize = 20)
    plt.yticks(fontweight = 'normal',fontsize = 20)

    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='major', direction='in', length=10, width=1)
    axes.yaxis.set_tick_params(which='minor', direction='in', length=5, width=1)

    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='major', direction='in', length=10, width=1)
    axes.xaxis.set_tick_params(which='minor', direction='in', length=5, width=1)

    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()


    handles = handles[::-1]
    handles = handles+handles
    handles = handles[1:12]

    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles)

    plt.ylabel("Events",fontsize = 20,fontweight='normal')
    axes.yaxis.set_label_coords(-0.07,0.93)
    label = variable
    if variable == 'mBB':
        label = r"$m_{bb}$ GeV"
    elif variable == 'Mtop':
        label = r"$m_{top}$ GeV"

    plt.xlabel(label,fontsize = 20,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)

    plt.show()


def sensitivity_cut_based(df):
    """Calculate sensitivity from dataframe with error"""

    # Initialise sensitivity and error.
    sens_sq = 0

    #Split into signal and background events
    classes = df['Class']
    dec_vals = df['mBB']
    weights = df['EventWeight']

    y_data = zip(classes, dec_vals, weights)

    events_sb = [[a[1] for a in deepcopy(y_data) if a[0] == 1], [a[1] for a in deepcopy(y_data) if a[0] == 0]]
    weights_sb = [[a[2] for a in deepcopy(y_data) if a[0] == 1], [a[2] for a in deepcopy(y_data) if a[0] == 0]]

    #plots histogram with optimised bins and counts number of signal and background events in each bin
    plt.ioff()
    counts_sb = plt.hist(events_sb,
                         bins=20,
                         weights=weights_sb)[0]
    plt.close()
    plt.ion()

    # Reverse the counts before calculating.
    # Zip up S, B, DS and DB per bin.
    s_stack = counts_sb[0][::-1]   #counts height of signal in each bin from +1 to -1
    b_stack = counts_sb[1][::-1]    #counts height of bkground in each bin from +1 to -1


    for s, b in zip(s_stack, b_stack): #iterates through every bin
        this_sens = 2 * ((s + b) * math.log(1 + s / b) - s) #calcs sensivity for each bin
        if not math.isnan(this_sens):   #unless bin empty add this_sense to sens_sq total (sums each bin sensitivity)
            sens_sq += this_sens


    # Sqrt operations and error equation balancing.
    sens = math.sqrt(sens_sq)

    return sens



def sensitivity_bdt(df):
    """Calculate sensitivity from dataframe with error"""

    # Initialise sensitivity and error.
    sens_sq = 0

    #Split into signal and background events
    classes = df['Class']
    dec_vals = df['decision_value']
    weights = df['EventWeight']

    y_data = zip(classes, dec_vals, weights)

    events_sb = [[a[1] for a in deepcopy(y_data) if a[0] == 1], [a[1] for a in deepcopy(y_data) if a[0] == 0]]
    weights_sb = [[a[2] for a in deepcopy(y_data) if a[0] == 1], [a[2] for a in deepcopy(y_data) if a[0] == 0]]

    #plots histogram with optimised bins and counts number of signal and background events in each bin
    plt.ioff()
    counts_sb = plt.hist(events_sb,
                         bins=20,
                         weights=weights_sb)[0]
    plt.close()
    plt.ion()

    # Reverse the counts before calculating.
    # Zip up S, B, DS and DB per bin.
    s_stack = counts_sb[0][::-1]   #counts height of signal in each bin from +1 to -1
    b_stack = counts_sb[1][::-1]    #counts height of bkground in each bin from +1 to -1


    for s, b in zip(s_stack, b_stack): #iterates through every bin
        this_sens = 2 * ((s + b) * math.log(1 + s / b) - s) #calcs sensivity for each bin
        if not math.isnan(this_sens):   #unless bin empty add this_sense to sens_sq total (sums each bin sensitivity)
            sens_sq += this_sens


    # Sqrt operations and error equation balancing.
    sens = math.sqrt(sens_sq)

    return sens
