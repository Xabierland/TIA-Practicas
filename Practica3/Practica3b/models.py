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
        # Initialize your model parameters here
        # For example:
        self.batch_size = 4
        # Layer 0
        self.w0 = nn.Parameter(1, 200)
        self.b0 = nn.Parameter(1, 200)
        # Layer 1
        self.w1 = nn.Parameter(200, 200)
        self.b1 = nn.Parameter(1, 200)
        # Layer 2
        self.w2 = nn.Parameter(200, 200)
        self.b2 = nn.Parameter(1, 200)
        # Layer 3
        self.w3 = nn.Parameter(200, 200)
        self.b3 = nn.Parameter(1, 200)
        # Layer 4
        self.w4 = nn.Parameter(200, 1)
        self.b4 = nn.Parameter(1, 1)
        # Learning rate
        self.lr = -0.009
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1). En este caso cada ejemplo solo esta compuesto por un rasgo
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values.
            Como es un modelo de regresion, cada valor y tambien tendra un unico valor
        """
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.w1), self.b1))
        layer3 = nn.ReLU(nn.AddBias(nn.Linear(layer2, self.w2), self.b2))
        layer4 = nn.ReLU(nn.AddBias(nn.Linear(layer3, self.w3), self.b3))
        return nn.AddBias(nn.Linear(layer4, self.w4), self.b4)
        
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
        total_loss = 100000
        while True:
            total_loss = 0
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                total_loss = nn.as_scalar(loss)
                grad_wrt_w0, grad_wrt_b0, grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2, grad_wrt_w3, grad_wrt_b3, grad_wrt_w4, grad_wrt_b4 = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4])
                self.w0.update(grad_wrt_w0, self.lr)
                self.b0.update(grad_wrt_b0, self.lr)
                self.w1.update(grad_wrt_w1, self.lr)
                self.b1.update(grad_wrt_b1, self.lr)
                self.w2.update(grad_wrt_w2, self.lr)
                self.b2.update(grad_wrt_b2, self.lr)
                self.w3.update(grad_wrt_w3, self.lr)
                self.b3.update(grad_wrt_b3, self.lr)
                self.w4.update(grad_wrt_w4, self.lr)
                self.b4.update(grad_wrt_b4, self.lr)
            
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

        output_size = 10 # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 CLASES
        pixel_dim_size = 28
        pixel_vector_length = pixel_dim_size* pixel_dim_size
 
        "*** YOUR CODE HERE ***"

     

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
            output_size = 10 # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 "COSENOS"
        """
        "*** YOUR CODE HERE ***"





    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        POR EJEMPLO: [0,0,0,0,0,1,0,0,0,0,0] seria la y correspondiente al 5
                     [0,1,0,0,0,0,0,0,0,0,0] seria la y correspondiente al 1

        EN ESTE CASO ESTAMOS HABLANDO DE MULTICLASS, ASI QUE TIENES QUE CALCULAR 
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"#NO ES NECESARIO QUE LO IMPLEMENTEIS, SE OS DA HECHO
        return nn.SoftmaxLoss(self.run(x), y) # COMO VEIS LLAMA AL RUN PARA OBTENER POR CADA BATCH
                                              # LOS 10 VALORES DEL "COSENO". TENIENDO EL Y REAL POR CADA EJEMPLO
                                              # APLICA SOFTMAX PARA CALCULAR LA PROBABILIDA MAX
                                              # Y ESA SERA SU PREDICCION,
                                              # LA CLASE QUE MUESTRE EL MAYOR PROBABILIDAD, LA PREDICCION MAS PROBABLE, Y LUEGO LA COMPARARA CON Y 

    def train(self, dataset):
        """
        Trains the model.
        EN ESTE CASO EN VEZ DE PARAR CUANDO EL ERROR SEA MENOR QUE UN VALOR O NO HAYA ERROR (CONVERGENCIA),
        SE PUEDE HACER ALGO SIMILAR QUE ES EN NUMERO DE ACIERTOS. EL VALIDATION ACCURACY
        NO LO TENEIS QUE IMPLEMENTAR, PERO SABED QUE EMPLEA EL RESULTADO DEL SOFTMAX PARA CALCULAR
        EL NUM DE EJEMPLOS DEL TRAIN QUE SE HAN CLASIFICADO CORRECTAMENTE 
        """
        batch_size = self.batch_size
        while dataset.get_validation_accuracy() < 0.97:
            #ITERAR SOBRE EL TRAIN EN LOTES MARCADOS POR EL BATCH SIZE COMO HABEIS HECHO EN LOS OTROS EJERCICIOS
            #ACTUALIZAR LOS PESOS EN BASE AL ERROR loss = self.get_loss(x, y) QUE RECORDAD QUE GENERA
            #UNA FUNCION DE LA LA CUAL SE  PUEDE CALCULAR LA DERIVADA (GRADIENTE)
            "*** YOUR CODE HERE ***"




