import nn

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    NO ES CLASIFICACION, ES REGRESION. ES DECIR; APRENDER UNA FUNCION.
    SI ME DAN X TENGO QUE APRENDER A OBTENER LA MISMA Y QUE EN LA FUNCION ORIGINAL DE LA QUE QUIERO APRENDER
    """
    def __init__(self):
        # Tamaño del batch
        self.batch_size = 4
        # Layer 0
        self.w0 = nn.Parameter(1, 5)
        self.b0 = nn.Parameter(1, 5)
        # Layer 1
        self.w1 = nn.Parameter(5, 5)
        self.b1 = nn.Parameter(1, 5)
        # Layer 2
        self.w2 = nn.Parameter(5, 1)
        self.b2 = nn.Parameter(1, 1)
        # Learning rate
        self.lr = -0.01
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1). En este caso cada ejemplo solo esta compuesto por un rasgo
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values.
            Como es un modelo de regresion, cada valor y tambien tendra un unico valor
        """
        layer0 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0))
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(layer0, self.w1), self.b1))
        return nn.AddBias(nn.Linear(layer1, self.w2), self.b2)
        
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
                ----> ES FACIL COPIA Y PEGA ESTO Y ANNADE LA VARIABLE QUE HACE FALTA PARA CALCULAR EL ERROR 
                return nn.SquareLoss(self.run(x),ANNADE LA VARIABLE QUE ES NECESARIA AQUI), para medir el error, necesitas comparar el resultado de tu prediccion con .... que?
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        
        """
        
        batch_size = self.batch_size
        while True:
            total_loss = 0
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                total_loss = nn.as_scalar(loss)
                grad_wrt_w0, grad_wrt_b0, grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])
                self.w0.update(grad_wrt_w0, self.lr)
                self.b0.update(grad_wrt_b0, self.lr)
                self.w1.update(grad_wrt_w1, self.lr)
                self.b1.update(grad_wrt_b1, self.lr)
                self.w2.update(grad_wrt_w2, self.lr)
                self.b2.update(grad_wrt_b2, self.lr)
            
            if total_loss < 0.02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        # TEN ENCUENTA QUE TIENES 10 CLASES, ASI QUE LA ULTIMA CAPA TENDRA UNA SALIDA DE 10 VALORES,
        # UN VALOR POR CADA CLASE

        # Tamaño del batch
        self.batch_size = 10
        
        # Learning rate
        self.lr = -0.01

        # Tamaño de salida
        output_size = 10
        
        # Dimensiones de la imagen
        pixel_vector_length = 28 * 28
 
        # Inicializa los pesos y sesgos
        # Layer 0
        self.w0 = nn.Parameter(pixel_vector_length, 100)
        self.b0 = nn.Parameter(1, 100)
        # Layer 1
        self.w1 = nn.Parameter(100, 100)
        self.b1 = nn.Parameter(1, 100)
        # Layer 2
        self.w2 = nn.Parameter(100, output_size)
        self.b2 = nn.Parameter(1, output_size)

    def run(self, x):
        """
        Corre el modelo para un lote de ejemplos.
        
        Inputs:
            x: un nodo con forma (batch_size x 784)
        Returns:
            Un nodo con forma (batch_size x 10) que contiene los valores predichos de y.
        """
        layer0 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0))
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(layer0, self.w1), self.b1))
        return nn.AddBias(nn.Linear(layer1, self.w2), self.b2)

    def get_loss(self, x, y):
        """
        Calcula la pérdida para un lote de ejemplos.
        
        Inputs:
            x: un nodo con forma (batch_size x 784) que se mete en la red para obtener las predicciones.
            y: un nodo con forma (batch_size x 10), que contiene los verdaderos valores y que se utilizarán para el entrenamiento.
        Returns: un nodo de pérdida
        """
        return nn.SoftmaxLoss(self.run(x), y) 
    
    def train(self, dataset):
        """
        Trains the model.
        EN ESTE CASO EN VEZ DE PARAR CUANDO EL ERROR SEA MENOR QUE UN VALOR O NO HAYA ERROR (CONVERGENCIA),
        SE PUEDE HACER ALGO SIMILAR QUE ES EN NUMERO DE ACIERTOS. EL VALIDATION ACCURACY
        NO LO TENEIS QUE IMPLEMENTAR, PERO SABED QUE EMPLEA EL RESULTADO DEL SOFTMAX PARA CALCULAR
        EL NUM DE EJEMPLOS DEL TRAIN QUE SE HAN CLASIFICADO CORRECTAMENTE 
        """
        while dataset.get_validation_accuracy() < 0.97:
            # Iterar sobre el dataset en lotes.
            for x, y in dataset.iterate_once(self.batch_size):
                # Calcula la pérdida.
                loss = self.get_loss(x, y)

                # Calcula el gradiente de los pesos y sesgos con respecto a la pérdida.
                gradients = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])

                # Actualiza los pesos y sesgos usando gradiente descendente.
                self.w0.update(gradients[0], self.lr)
                self.b0.update(gradients[1], self.lr)
                self.w1.update(gradients[2], self.lr)
                self.b1.update(gradients[3], self.lr)
                self.w2.update(gradients[4], self.lr)
                self.b2.update(gradients[5], self.lr)
