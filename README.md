## Image-Segmentation using U-Net archictecture 
### Dataset
`lyft-udacity-challenge` dataset, sourced from Kaggle <br>
Containing 5000 images. 

### Process the data 
```
def preprocess(image, mask):
  input_image = tf.image.resize(image, IMG_SHAPE, method = 'nearest')
  input_mask = tf.image.resize(mask, IMG_SHAPE, method = 'nearest')

  return input_image, input_mask

def process_path(image_path, mask_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_png(img, channels = 3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  mask = tf.io.read_file(mask_path)
  mask = tf.image.decode_png(mask, channels = 3)
  mask = tf.math.reduce_max(mask, axis = -1, keepdims = True) #mask class is determined 

  img, mask = preprocess(img, mask,)

  return img, mask
  ```
  ### Unet 
- U-Net improves on the FCN differing in some important ways. 
- Instead of one transposed convolution at the end of the network, it uses a matching number of convolutions for downsampling the input image to a feature map, and transposed convolutions for upsampling those maps back up to the original input image size. 
- It also adds skip connections, to retain information that would otherwise become lost during encoding. 
- Skip connections send information to every upsampling layer in the decoder from the corresponding downsampling layer in the encoder, capturing finer information while also keeping computation low. These help prevent information loss, as well as model overfitting.


  ## Modelling 
Defined `conv_block`, `upsampling block`, `unet_model`
  ### `conv_block`
The encoder is a stack of various conv_blocks:
- Each `conv_block()` is composed of `2 Conv2D` layers with `ReLU` activations. We will apply Dropout,and MaxPooling2D to some conv_blocks, specifically to the last two blocks of the downsampling.
- The function will return two tensors:
  - `next_layer`: That will go into the next block.
  - `skip_connection`: That will go into the corresponding decoding block.
**Note**: If max_pooling=True, the next_layer will be the output of the MaxPooling2D layer, but the skip_connection will be the output of the previously applied layer(Conv2D or Dropout, depending on the case). Else, both results will be identical.
  ### `upsampling block`
  - The decoder, or upsampling block, upsamples the features back to the original image size. At each upsampling level, we'll take the output of the corresponding encoder block and concatenate it before feeding to the next decoder block.
  - 2 new components in the decoder: `up and merge`. These are the transpose convolution and the skip connections. 
  - `Conv2DTranspose` layer, performs the inverse of the Conv2D layer
 ### `unet_model`
 This is where you'll put it all together, by chaining the encoder, bottleneck, and decoder! we need to specify the number of output channels, which for this particular set would be 13. 
 
  
  ### Evaluation on test set 
  ```
  125/125 [==============================] - 105s 25ms/step - loss: 0.1630 - accuracy: 0.9489
```

### Predictions 
<img width="420" alt="Screenshot 2023-05-21 at 10 35 46 AM" src="https://github.com/ayushs0911/Image-Segmentation/assets/122048067/cdfdb3b8-e856-4dc4-aa12-2bcad444190d">
