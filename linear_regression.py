import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression

class linear_regression:
    def __init__(self,x,y):
        self.x=np.array(x)
        self.y=np.array(y)
        self.x_mean=np.mean(x)
        self.y_mean=np.mean(y)
        self.w=np.random.randn()
        self.b=np.random.randn()
        self.find_fx()

    def find_fx(self):
        """
        :param x: x_mean, y_mean, x[i], y[i]
        :math fx: w=E((x[i]-x_mean)*(y[i]-y_mean))/E((x[i]-x_mean)^2), E: tổng xích ma
                : b=y_mean-w*x_mean
        :return: w và b (y=w*x+b)
        """

        self.w=np.sum((self.x-self.x_mean)*(self.y-self.y_mean))/np.sum((self.x-self.x_mean)**2)
        self.b=self.y_mean-self.w*self.x_mean
        return self.w,self.b

    def predict(self,x_need_find):
        return self.w*x_need_find+self.b

    def plot_fx(self,x_need_find=0.001):
        fig,ax=plt.subplots(figsize=(5,5))
        x_plot=np.linspace(min(self.x)-1,max(self.x)+1,10)
        y_plot=self.predict(x_plot)
        ax.plot(x_plot,y_plot,color="r")
        ax.set_title("Linear Regression")
        ax.scatter(x,y,color='g')
        x_need_find=np.array(x_need_find)
        y_need_find=self.predict(x_need_find)
        ax.scatter(x_need_find,y_need_find,color="b")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(["f(x)","data point","pred"])
        ax.grid()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))


class Gadient_Descent_Linear_Regression:
    def __init__(self,x,y,learning_rate=0.01,epochs=100):
        self.x=np.array(x)
        self.y=np.array(y)
        np.random.seed(42)
        self.w=np.random.randn()
        self.b=np.random.rand()
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.find_line()

    def find_line(self):
        """
        :param x:
        :math fx:
                :
        :return:
        """
        for epoch in range(self.epochs):
            self.w-=self.learning_rate*(-2/len(self.x))*(np.sum(self.x*(self.y-(self.w*self.x+self.b))))
            self.b-=self.learning_rate*(-2/len(self.x))*(np.sum(self.y-(self.w*self.x+self.b)))
        return self.w, self.b

    def predict(self,x_need_find):
        return self.w*x_need_find+self.b

    def plot_line_GD(self,x_need_find=0.001):
        fig,ax=plt.subplots(figsize=(5,5))
        x_plot=np.linspace(min(self.x)-1,max(self.x)+1,10)
        y_plot=self.predict(x_plot)
        ax.plot(x_plot,y_plot,color="r")
        ax.set_title("Linear Regression Gadient Descent")
        ax.scatter(x,y,color='g')
        x_need_find=np.array(x_need_find)
        y_need_find=self.predict(x_need_find)
        ax.scatter(x_need_find,y_need_find,color="b")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(["f(x)","data point","pred"])
        ax.grid()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))


def sk(x,y,x_need_find):
    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)
    x_need_find = np.array(x_need_find).reshape(-1, 1)
    y_need_find=model.predict(x_need_find)

    print(f"Sklearn Linear Regression Predict:          x= {x_need_find.item()} => y= {y_need_find.item()}")

    fig, ax = plt.subplots( figsize=(5, 5))
    ax.set_title('Linear Regression Sklearn')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x, y_pred, color='red')
    ax.scatter(x, y, color='g')
    ax.scatter(x_need_find, y_need_find, color='b')
    ax.grid(True)
    ax.legend(["f(x)","data point","pred"])
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))


def main():
    #Normal Equation linear_regression
    linear_regression_pred=linear_regression(x,y)
    x_need_find=11
    print(f"Linear Regression Predict:                  x= {x_need_find} => y= {linear_regression_pred.predict(x_need_find)}")
    linear_regression_pred.plot_fx(x_need_find)
    #Gadient Descent linear_regression
    GD_Linear_Regression_pred=Gadient_Descent_Linear_Regression(x,y,learning_rate=0.01,epochs=1000)
    print(f"Gadient Descent Linear Regression Predict:  x= {x_need_find} => y= {GD_Linear_Regression_pred.predict(x_need_find)}")
    GD_Linear_Regression_pred.plot_line_GD(x_need_find)
    #linear_regression from scikit-learn
    sk(x, y, x_need_find)

if __name__=="__main__":
    data = np.array([[1,2],
                     [3,3],
                     [4,3.5],
                     [5,5],
                     [7,7],
                     [8,9],
                     [10,8],
                     [12,11],
                     [3,3.5],
                     [2,1],
                     [10,9],
                     [7,5]])
    x = data[:, 0].reshape(-1,1)
    y = data[:, 1].reshape(-1,1)
    main()
    plt.show()