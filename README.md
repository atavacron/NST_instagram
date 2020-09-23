# Connecting with Google Drive
We are going to use Google Drive API. So we have to enable it. We want it to be a Desktop app. Once done that we can download the credentials.json. This one includes the Client ID and Client Secret.

## Neural Algorithm of Artistic Style
In order to understand neural style transfer we are going to use Gatys, Ecker & Bethge (2015) paper. For image processing we are going to use Convolutional Neural Networks (CNN), a class of Deep Neural Networks. CNN is about processing small units of visual information in a feed-forward manner. They are basically image filters, that extract specific characteristics of the given image. We can extract the information that each layer contains. Higher layers capture high-level content called **content representation**.<br>
For the style we have to use a feature space designed to cature TEXTURE inforamtion, built on top filter responses in each layer, and then computing the correlations. These feature correlations are given by a gram matrix. This paper uses 5 layers from the VGG architecture for the content characteristics and from style iterates over one to 5 layers. 
### Content and Style is separable in Convolutional Neural Networks.
The loss function contains a term for content and a term for style. Strng style importance will care on transfering style while strong content will care on keeping content, the point is to find the trade-off or ratio alpha/beta, that are the weights for the loss of the content and the loss of the style respectively. To generate the texture that matches the style gradient descent has been used by minimizing the mean square distance between the original gram matrix and the generated image gram matrix. 
For more information visit: https://arxiv.org/pdf/1508.06576.pdf
