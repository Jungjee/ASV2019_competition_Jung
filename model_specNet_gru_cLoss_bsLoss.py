import numpy as np
import keras
import tensorflow as tf
from keras import regularizers, optimizers, utils, models, initializers, constraints
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Activation, Input, Add, Dropout, LeakyReLU, GRU, AveragePooling2D, Concatenate, Reshape, Lambda
import keras.backend as K 
from keras.models import Model
from keras.engine.topology import Layer
from keras.activations import softmax


import os
_abspath = os.path.abspath(__file__)
m_name = _abspath.split('/')[-1].split('.')[0][6:]

def squeeze_middle2axes_operator( x4d ) :
    shape = tf.shape( x4d ) # get dynamic tensor shape
    x3d = tf.reshape( x4d, [shape[0], shape[1] * shape[2], shape[3] ] )
    return x3d

def squeeze_middle2axes_shape( x4d_shape ) :
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if ( None in [ in_rows, in_cols] ) :
        output_shape = ( in_batch, None, in_filters )
    else :
        output_shape = ( in_batch, in_rows * in_cols, in_filters )
    return output_shape

'''
# define a dummy model
xx = Input( shape=(None, None, 16) )
yy = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( xx )
mm = Model( inputs = xx, outputs = yy )
print mm.summary()
'''

def simple_loss(y_true, y_pred):
	return K.mean(y_pred)


def zero_loss(y_true, y_pred):
	return 0.5 * K.sum(y_pred, axis=0)


def residual_block(l, increase_dim = False, first = False, keep_filter = False, strides = None, r_value = None):
	input_num_filters = keras.backend.int_shape(l)[-1]
	
	if increase_dim:
		first_stride = strides
		if keep_filter:
			out_num_filters = input_num_filters

		else:
			out_num_filters = input_num_filters * 2
	else:
		first_stride = (1, 1)
		out_num_filters = input_num_filters
		
	if first:
		bn_pre_relu = l
	if not first:
		bn_pre_conv = BatchNormalization()(l)
		bn_pre_relu = LeakyReLU()(bn_pre_conv)
	
	conv_1 = LeakyReLU()(
					BatchNormalization()(
						Conv2D(out_num_filters, (3, 5),strides = first_stride, padding ='same',  kernel_regularizer=regularizers.l2(r_value))(bn_pre_relu)
					)
				)
	conv_2 = Conv2D(out_num_filters, (3, 5), padding ='same',  kernel_regularizer=regularizers.l2(r_value))(conv_1)
		
	if increase_dim:
		projection = Conv2D(out_num_filters, first_stride, strides = first_stride, padding ='same', use_bias=False,  kernel_regularizer=regularizers.l2(r_value))(l)
		#projection = MaxPooling2D(pool_size=first_stride, padding ='same')(l)
		#projection = Conv2D(out_num_filters, (1,1), strides = (1,1), padding ='same', use_bias=False,  kernel_regularizer=regularizers.l2(r_value))(projection)
		
		block = Add()([conv_2 , projection])
	else:
		block = Add()([conv_2 , l])

	return block


class get_output_layer(Dense):
	def __init__(self, kernel, **kwargs):
		super(Dense, self).__init__(**kwargs)
		self.kernel = kernel
	def build(self, input_shape):
		self.built = True
	def call(self, inputs):
		inner_output = K.dot(inputs, self.kernel)
		softmax_output = softmax(inner_output)
		
		return softmax_output
		
	def compute_output_shape(self, input_shape):
		return (input_shape[0], K.int_shape(self.kernel)[-1])


class spk_basis_loss(Dense):
	def __init__(self, units, with_H = False, s = 5., negative_k = 100, num_batch = 100,
				 kernel_initializer='glorot_uniform',
				 kernel_regularizer=None,
				 kernel_constraint=None,
				 **kwargs):
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs.pop('input_dim'),)
		super(Dense, self).__init__(**kwargs)
		self.units = units
		
		self.with_H = with_H
		self.s = s
		self.negative_k = negative_k
		self.num_batch = num_batch
		
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		

	def build(self, input_shape):
		assert len(input_shape[0]) >= 2
		input_dim = input_shape[0][-1]

		self.kernel = self.add_weight(shape=(input_dim, self.units),
									  initializer=self.kernel_initializer,
									  name='kernel',
									  regularizer=self.kernel_regularizer,
									  constraint=self.kernel_constraint)
		self.bias = None
		self.built = True


	def call(self, inputs):
		inputs_x = inputs[0]
		inputs_y = inputs[1]
		
		input_length = K.sum(inputs_x**2., axis = 1, keepdims = True)**0.5
		input_length /= self.s ** 0.5
		input_length += 0.0001
		
		kernel_length = K.sum(self.kernel**2., axis = 0, keepdims = True)**0.5
		kernel_length /= self.s ** 0.5
		kernel_length += 0.0001
		
		inputs_norm = inputs_x / input_length
		kernel_norm = self.kernel / kernel_length
		
		#label_onehot = tf.one_hot(tf.reshape(inputs_y, [-1]), self.units)
		label_onehot = inputs_y
		# shape = [#batch_sample, #spk]
		
		negative_mask = tf.fill([self.units, self.units], 1.) - tf.eye(self.units)
		# shape = [#spk, #spk]
		
		negative_mask2 = tf.fill([self.num_batch, self.units], 1.) - label_onehot
		# shape = [#batch_sample, #spk]
		
		loss_BS = K.mean(tf.matmul(kernel_norm, kernel_norm,
                     adjoint_a = True # transpose second matrix
                     ) * negative_mask  ) 
					 
		if self.with_H:		
			cos_output = K.dot(inputs_norm, kernel_norm)	
			cos_target = K.sum(cos_output * label_onehot, axis = 1, keepdims = True)
			
			
			cos_diff = K.exp(cos_output - cos_target) * negative_mask2
			hard_negatives, _ = tf.nn.top_k(cos_diff, k=self.negative_k,sorted=False)
			
			loss_H = K.mean(K.log(1. + hard_negatives), axis = 1)
			
			final_loss = loss_H + loss_BS
		else:
			
			inner_output = K.dot(inputs_x, self.kernel)
			softmax_output = softmax(inner_output)
			#loss_s = K.sparse_categorical_crossentropy(inputs_y, softmax_output)
			loss_s = K.categorical_crossentropy(inputs_y, softmax_output)
			
			final_loss = loss_s + loss_BS
		
		
		return final_loss
		
		
	def compute_output_shape(self, input_shape):
		return (input_shape[0][0], 1)


class CenterLossLayer(Layer):

	def __init__(self, alpha, nb_center, dim_embd, **kwargs):
		super().__init__(**kwargs)
		self.alpha = alpha
		self.nb_center = nb_center
		self.dim_embd = dim_embd

	def build(self, input_shape):
		self.centers = self.add_weight(name='centers',
				   shape=(self.nb_center, self.dim_embd),
				   initializer='uniform',
				   trainable=False)
		# self.counter = self.add_weight(name='counter',
		#			shape=(1,),
		#			initializer='zeros',
		#			trainable=False)  # just for debugging
		super().build(input_shape)

	def call(self, x, mask=None):

		# x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
		delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
		center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
		delta_centers /= center_counts
		new_centers = self.centers - self.alpha * delta_centers
		self.add_update((self.centers, new_centers), x)

		# self.add_update((self.counter, self.counter + 1), x)

		self.result = x[0] - K.dot(x[1], self.centers)
		self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
		return self.result # Nx1

	def compute_output_shape(self, input_shape):
		return K.int_shape(self.result)

	
def get_model(argDic):
	#(60, 1025, 2)
	#inputs = Input(shape=(None, 1025, 1), name = 'input_gru')
	#inputs = Input(shape=(None, 1025, 2), name = 'input_gru')
	inputs = Input(shape=(None, 1025, 3), name = 'input_gru')
	c_input = Input(shape = (argDic['nb_spk'],))
	
	l = Conv2D(16, (3, 7),strides=(1,1), padding = 'same',  kernel_regularizer=regularizers.l2(argDic['wd']))(inputs)
	l = BatchNormalization()(l)
	l = LeakyReLU()(l)
	
	l = residual_block(l, first=True, r_value = argDic['wd'])
	for _ in range(1,3):
		l = residual_block(l, r_value = argDic['wd'])
	print (keras.backend.int_shape(l))
	
	l = residual_block(l, increase_dim=True, strides = (2,4), r_value = argDic['wd'])
	for _ in range(1,4):
		l = residual_block(l, r_value = argDic['wd'])
	print (keras.backend.int_shape(l))
	
	l = residual_block(l, increase_dim=True, strides = (2,4), r_value = argDic['wd'])
	for _ in range(1,4):
		l = residual_block(l, r_value = argDic['wd'])
	print (keras.backend.int_shape(l))
	
	l = residual_block(l, increase_dim=True, keep_filter = False, strides = (2,4), r_value = argDic['wd'])
	for _ in range(1,3):
		l = residual_block(l, r_value = argDic['wd'])
	print (keras.backend.int_shape(l))
	
	
	l = BatchNormalization()(l)
	l = LeakyReLU()(l)
	print (keras.backend.int_shape(l))
	x_m = MaxPooling2D((1, 17))(l)
	x_a = AveragePooling2D((1, 17))(l)
	x = Concatenate()([x_m, x_a])
	#x = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )(x)
	#x = Reshape((keras.backend.int_shape(x)[1], 256))(x)
	x = Reshape((-1, 256))(x)
	print (keras.backend.int_shape(x))

	for i in range(0, len(argDic['nb_gru_node'])):
		r_seq = False if i == len(argDic['nb_gru_node']) -1 else True
		x = GRU(argDic['nb_gru_node'][i],
			activation='tanh',
			recurrent_activation='hard_sigmoid',
			kernel_initializer='glorot_uniform',
			recurrent_initializer='orthogonal',
			kernel_regularizer=regularizers.l2(argDic['wd']),
			recurrent_regularizer=regularizers.l2(argDic['wd']),
			dropout=0.0,
			recurrent_dropout=argDic['req_drop'],
			implementation=1,
			return_sequences= r_seq,
			go_backwards=False,
			reset_after=False,
			name = 'gru_%d'%i)(x)

	for i in range(len(argDic['nb_dense_node'])):
		if i == len(argDic['nb_dense_node']) -1:
			name = 'gru_code'
		else:
			name = 'gru_dense_act_%d'%(i+1)
		x = Dense(argDic['nb_dense_node'][i],
			kernel_initializer = argDic['initializer'],
			kernel_regularizer = regularizers.l2(argDic['wd']),
			name = 'gru_dense_%d'%i)(x)
		x = BatchNormalization(axis=-1, name='gru_BN_%d'%i)(x)
		x = LeakyReLU(name = name)(x)
	
	s_bs_layer = spk_basis_loss(with_H = bool(argDic['use_H_loss']),
			units = argDic['nb_spk'],
			negative_k = argDic['bs_nb_k'],
			kernel_initializer = argDic['initializer'],
			kernel_regularizer = regularizers.l2(argDic['wd']),
			num_batch = argDic['batch_size'],
			name = 'gru_s_bs_loss')

	s_bs_out = s_bs_layer([x, c_input])

	output_layer = get_output_layer(kernel = s_bs_layer.kernel, name = 'softmax_output')
	softmax_output = output_layer(x)

	c_out = CenterLossLayer(alpha = argDic['c_alpha'],
			nb_center = argDic['nb_spk'],
			dim_embd = argDic['nb_dense_node'][-1],
			name='gru_c_loss')([x, c_input])
	return [Model(inputs=[inputs, c_input], output=[s_bs_out, c_out]), m_name, softmax_output]










