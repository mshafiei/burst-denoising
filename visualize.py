import numpy as np
import os
import pickle
import tensorflow as tf
import io
from tensorboardX import SummaryWriter
import subprocess
import datetime
from natsort import natsorted
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import time
def visualize(inpt):

    # g = self.quad_model(inpt['net_input'])
    g,fft_res = inpt['g'],inpt['fft_res']
    gx = g[...,:3]
    gy = g[...,3:]
    # predict = tfu.camera_to_rgb_batch(predict/inpt['alpha'],inpt)
    dx = np.roll(fft_res, 1, axis=[2]) - fft_res
    dy = np.roll(fft_res, 1, axis=[1]) - fft_res

    dxx = np.roll(dx, 1, axis=[2]) - dx
    dyy = np.roll(dy, 1, axis=[1]) - dy
    gxx = np.roll(gx, 1, axis=[2]) - gx
    gyy = np.roll(gy, 1, axis=[1]) - gy
    loss_data = ((dx - gx) ** 2 + (dy - gy) ** 2).mean()
    loss_smoothness = ((fft_res/inpt['alpha'] - inpt['preambient']) ** 2).mean()
    
    
    
    out = [inpt['predict'],inpt['ambient'],inpt['noisy'],fft_res/inpt['alpha'],
            np.abs(gxx)*1000,np.abs(dxx/inpt['alpha'])*100,
            np.abs(gyy)*1000,np.abs(dyy/inpt['alpha'])*100,
            np.abs(gx),np.abs(dx/inpt['alpha'])*100,
            np.abs(gy),np.abs(dy/inpt['alpha'])*100]
    return out,{'loss_data':loss_data,'loss_smoothness':loss_smoothness}
    
def labels(mtrcs_pred,mtrcs_inpt):
    out = [r'$Prediction~PSNR~%.3f$'%mtrcs_pred['psnr'][0],
           r'$Ground Truth$',r'$I_{noisy}~PSNR:%.3f$'%mtrcs_inpt['psnr'][0],
           r'$I$',r'$Unet~output~(|g^x_x|)~\times~1000$',r'$|I_{xx}|~\times~100$',r'$Unet~output~(|g^y_y|)~\times~1000$',r'$|I_{yy}|~\times~100$',
    r'$Unet~output~(|g^x|)~\times~1.$',r'$|I_{x}|~\times~100$',r'$Unet~output (|g^y|)~\times~1.$',r'$|I_{y}|~\times~100$']
    return out

# #copied from https://stackoverflow.com/questions/42142144/displaying-first-decimal-digit-in-scientific-notation-in-matplotlib
# class ScalarFormatterForceFormat(ScalarFormatter):
#     def _set_format(self):  # Override function that finds format to use.
#         self.format = "%1.1f"  # Give format here

def getPathFilename(fn):
    bn = os.path.basename(fn)
    return fn[:-len(bn)]


def createIfNExist(dr):
    if(not os.path.exists(dr)):
        os.makedirs(dr)

def savePickle(relfn,obj):
    #create parent path if not exist
    fn = os.path.abspath(relfn)
    parentPath = getPathFilename(fn)
    createIfNExist(parentPath)
    with open(fn,'wb') as fd:
        pickle.dump(obj,fd)

def loadPickle(fn):
    with open(fn,'rb') as fd:
        obj = pickle.load(fd)
    return obj
def get_psnr(pred, gt):
    pred = tf.clip_by_value(pred, 0., 1.)
    gt = tf.clip_by_value(gt, 0., 1.)
    mse = tf.reduce_mean((pred - gt)**2.0, axis=[1, 2, 3])
    psnr = tf.reduce_mean(-10. * tf.log(mse) / tf.log(10.))
    return psnr
def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return (np.asarray(img) / 255.).astype(np.float32)
class logger:
    def __init__(self,opts,info=None):
        load_params, store_params = opts.load_params, opts.store_params
        if(load_params):
            expName = opts.expname
            path=opts.logdir
            params_fn = os.path.join(path,expName,'./app_params.pickle')
            opts = loadPickle(params_fn)
            print('loaded ',params_fn)

        self.opts = opts
        path=opts.logdir
        ltype=opts.logger
        projectName = opts.projectname
        expName = opts.expname

        self.path_train = os.path.join(path,expName,'train')
        self.path_val = os.path.join(path,expName,'val')
        self.path_test = os.path.join(path,expName,'test')
        print('log path: ',os.path.join(path,expName))
        if(ltype == 'tb' or ltype == 'filesystem'):
            createIfNExist(self.path_train)
            createIfNExist(self.path_val)
            createIfNExist(self.path_test)
        self.ltype = ltype
        self.step = 0
        self.writer_train = SummaryWriter(self.path_train)
        self.writer_val = SummaryWriter(self.path_val)
        self.writer_test = SummaryWriter(self.path_test)

        self.profiler_time = {}
        self.profiler_memory = {}

        if(store_params):
            params_fn = os.path.join(path,expName,'./app_params.pickle')
            savePickle(params_fn,opts)
            print('stored ',params_fn)
            exit(0)
                
        self.info = self.opts.__dict__
        if(self.info is not None):
            #additional info
            cmd = 'git rev-parse HEAD'
            head_id = subprocess.check_output(cmd, shell=True)
            readabletime = datetime.datetime.fromtimestamp(time.time()).strftime("%m/%d/%Y, %H:%M:%S")
            self.info.update({'head_id':head_id,'datetime':readabletime})
            self.addDict(self.info,'info')

    def addDict(self,info,label,mode='train'):
        for k,v in info.items():
            txt = k + '\t:\t' + str(v)
            self.addString(txt,label,mode)
            print(txt)

    @staticmethod
    def parse_arguments(parser):
        parser.add_argument('--logdir', type=str, default='./logger/Unet_test',help='Direction to store log used as ')
        parser.add_argument('--logger', type=str, default='tb',choices=['tb','filesystem'],help='Where to dump the logs')
        parser.add_argument('--expname', type=str, default='unvet_overfit_hardcode',help='Name of the experiment used as logdir/exp_name')
        parser.add_argument('--projectname', type=str, default='Unet_test',help='Name of the experiment used as logdir/exp_name')
        parser.add_argument('--store_params', action='store_true',help='Store parameters for debugging')
        parser.add_argument('--load_params', action='store_true',help='Load parameters for debugging')
        return parser

    def save_params(self,params,state,idx):
        dr = os.path.join(self.path_train,'params')
        if(not os.path.exists(dr)):
            os.makedirs(dr)
        fn = os.path.join(dr,'params_%i.pickle' % self.step)
        with open(fn,'wb') as fd:
            pickle.dump({'params':params,'state':state,'step':self.step,'idx':idx},fd)
        fn = os.path.join(dr,'latest_parameters.pickle')
        with open(fn,'wb') as fd:
            pickle.dump({'params':params,'state':state,'step':self.step,'idx':idx},fd)

    def load_params(self):
        dr = os.path.join(self.path_train,'params')
        if(not os.path.exists(dr) or not os.listdir(dr)):
            print('No params founds')
            return None
        else:
            fn = natsorted(os.listdir(dr))[-1]
            with open(os.path.join(dr,'latest_parameters.pickle'),'rb') as fd:
                data = pickle.load(fd)
                self.step = data['step']
                return data

    def tick(self,name):
        self.profiler_time[name] = time.time()

    def tock(self,name):
        self.profiler_time[name] = time.time() - self.profiler_time[name]


    def addImage(self,im,label,title,trnsfrm=lambda x:x,dim_type='HWC',mode='train'):
        """[Clips and shows a an image or a list of images]

        Args:
            im ([ndarray or list of ndarray]): []
            label ([str or list of str]): [description]
            dim_type ([str]): [arrangement of dimensions 'HWC' or 'BHWC' or 'CHW' or 'BCHW']
        """
        
        if(type(im) == list):
            if(dim_type == 'BCHW'):
                im = [trnsfrm(i).transpose([0,2,3,1]) for i in im]
            elif(dim_type == 'CHW'):
                im = [trnsfrm(i).transpose([1,2,0])[None,...] for i in im]
            elif(dim_type == 'HWC'):
                im = [trnsfrm(i)[None,...] for i in im]
            elif(dim_type == 'BHWC'):
                im = [trnsfrm(i) for i in im]
            im = [i for i in im]
            count, b, h, w, _ = len(im), im[0].shape[0],im[0].shape[1],im[0].shape[2],im[0].shape[3]
            fig = make_subplots(rows=b, cols=count,subplot_titles=label,horizontal_spacing=0,vertical_spacing=0)
            fig.update_yaxes(visible=False, showticklabels=False)
            fig.update_xaxes(visible=False, showticklabels=False)
            fig.update_layout(
                margin=dict(l=3, r=3, t=30, b=3),
            )
            [fig.add_trace(px.imshow(np.clip(im_j,0,1)).data[0], row=j+1, col=i+1)  for i,im_i in enumerate(im) for j,im_j in enumerate(im_i)]
            margin = 30
            fig.update_layout(height=h * b + margin, width=w*count + margin*2)
            im = plotly_fig2array(fig)


        writer, path = self.path_parse(mode)

        if(self.ltype == 'tb'):
            imshow = im[...,:3].transpose([2,0,1])
            writer.add_image(title.replace(' ','_'), imshow, self.step)
        elif(self.ltype == 'filesystem'):
            name = os.path.join(path,'%010i_%s.png' %(self.step, title.replace(' ','_')))
            imwrite(name,im[...,:3])

    def addScalar(self,scalar,label,mode='train',display_name=None):
        """[Adds a scalar to the logs for current time step]

        Args:
            scalar ([float,dict]): [Either add the float or every element in the dict]
            label ([str]): [Adds to the label]
        """
        writer, path = self.path_parse(mode)
        display_name = label if display_name is None else display_name
        if(type(scalar) == dict):
            for k,v in scalar.items():
                self.addScalar(v,k)
        elif(self.ltype == 'tb'):
            writer.add_scalar(label, float(scalar), self.step)
        elif(self.ltype == 'filesystem'):
            fn = os.path.join(path,label + '.txt')
            with open(fn, 'a') as fd:
                fd.write('%s : %f\n' % (label,scalar) )

    def addMetrics(self,mtrcs,mode='train'):
        for k,v in mtrcs.items():
            self.addScalar(v,k,mode=mode)

    def path_parse(self,mode='train'):
        if(mode == 'train'):
            writer = self.writer_train if(self.ltype == 'tb') else None
            path = self.path_train
        elif(mode == 'val'):
            writer = self.writer_val if(self.ltype == 'tb') else None
            path = self.path_val
        elif(mode == 'test'):
            writer = self.writer_test if(self.ltype == 'tb') else None
            path = self.path_test
        else:
            print('Invalid log mode')
        return writer, path

    def addString(self,text,label,mode='train'):
        _, path = self.path_parse(mode)
        fn = os.path.join(path,label + '.txt')
        # if(self.ltype == 'filesystem'):
        with open(fn, 'a') as fd:
            fd.write(text + '\n')



    def takeStep(self):
        if(self.ltype == 'tb'):
            self.writer_train.flush()
            self.writer_val.flush()
            self.writer_test.flush()
        self.step += 1
