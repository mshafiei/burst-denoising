import tensorflow as tf
import numpy as np
import math

def dx(x,method='bck'):
  if(method == 'bck'):
    y = np.pad(x,((0,0),(0,0),(1,0),(0,0)),mode='constant')
    return y[...,:,1:,:] - y[...,:,:-1,:]
  elif(method == 'fwd'):
    y = np.pad(x,((0,0),(0,0),(0,1),(0,0)),mode='constant')
    return y[...,:,:-1,:] - y[...,:,1:,:]
  else:
    print('Unrecognized derivative method')
    exit(0)

def dy(x,method='bck'):
  if(method == 'bck'):
    y = np.pad(x,((0,0),(1,0),(0,0),(0,0)),mode='constant')
    return y[...,1:,:,:] - y[...,:-1,:,:]
  elif(method == 'fwd'):
    y = np.pad(x,((0,0),(0,1),(0,0),(0,0)),mode='constant')
    return y[...,:-1,:,:] - y[...,1:,:,:]
  else:
    print('Unrecognized derivative method')
    exit(0)

def screen_poisson(lambda_d, img,grad_x,grad_y,IMSZ):
    img_freq = tf.signal.fft2d(tf.dtypes.complex(img,tf.zeros_like(img)))
    grad_x_freq = tf.signal.fft2d(tf.dtypes.complex(grad_x,tf.zeros_like(grad_x)))
    grad_y_freq = tf.signal.fft2d(tf.dtypes.complex(grad_y,tf.zeros_like(grad_y)))
    shape = [IMSZ,IMSZ]
    sx = np.fft.fftfreq(IMSZ).astype(np.float32)
    sx = np.repeat(sx, IMSZ)
    sx = np.reshape(sx, [IMSZ,IMSZ])
    sx = np.transpose(sx)
    sy = np.fft.fftfreq(IMSZ).astype(np.float32)
    sy = np.repeat(sy, IMSZ)
    sy = np.reshape(sy, shape)

    # Fourier transform of shift operators
    Dx_freq = 2 * math.pi * (np.exp(-1j * sx) - 1)
    Dy_freq = 2 * math.pi * (np.exp(-1j * sy) - 1)
    Dx_freq = tf.convert_to_tensor(Dx_freq)[None,None,...]
    Dy_freq = tf.convert_to_tensor(Dy_freq)[None,None,...]
    # my_grad_x_freq = Dx_freq * img_freqs)
    # my_grad_x_freq & my_grad_y_freq should be the same as grad_x_freq & grad_y_freq

    recon_freq = (tf.dtypes.complex(lambda_d,tf.zeros_like(lambda_d)) * img_freq + tf.math.conj(Dx_freq) * grad_x_freq + tf.math.conj(Dy_freq) * grad_y_freq) / \
                (tf.dtypes.complex(lambda_d,tf.zeros_like(lambda_d)) + (tf.math.conj(Dx_freq) * Dx_freq + tf.math.conj(Dy_freq) * Dy_freq))
    return tf.math.real(tf.signal.ifft2d(recon_freq))
