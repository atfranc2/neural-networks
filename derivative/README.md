# Derivatives

Derivatives are an important component of understanding how neural networks work because they are used in backpropagation to adjust neuron weights to minimize the loss function. 

A derivative is the slope or rate of change of a function at a certain point. The slope of the line is such that it is tangent to the function at that point. Mathematically, this is:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

So if we have a function $f(x) = 3x^2 + 4x + 6$ and we want to approximate the derivate at $x=2$, we would choose an h close to 0 such as $h=0.0001$.

$$
\frac{3(2+0.0001)^2 + 4(2 + 0.0001) + 6 - 3(2)^2 + 4(2) + 6}{0.0001} \approx 16
$$

So by using the approximation from limits we have $f'(2) = 16$. We can also use knowledge of calculus to know that in this case $f'(x) = 6x + 4$ and if we evaluate that we get the same outcome $f'(2) = 16$. 

![alt text](/derivative/figures/derivative_plots.png)

# Chain Rule

The chain rule is used to take the derivative of a composite function. This is a key concept in backpropagation because a neural network is essentially a composition of interconnected functions. So in order to minimize the loss function we have to take the derivative of the network function. 

The chain rule is defined using these rules: 

If $y = f(u)$ and $u = g(x)$ then $\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}$

This means that if we have a function $f(x) = x^7$ and another function $g(x) = 4*x^2 + 5*x + 6$ where $f(g(x)) = (4*x^2 + 5*x + 6)^7$

Then $f'(g(x)) = f'(u) * g'(x) = 7u^6 * (8x + 5) = 7(4*x^2 + 5*x + 6)^6 * (8x + 5)$

Note here that we only composed two functions but we can comose any number of functions (e.g $f(g(h(c(b))))$) 

# Resources
- [Derivative Mathematics](https://www.britannica.com/science/derivative-mathematics)
- [Chain Rule Video](https://youtu.be/H-ybCx8gt-8?si=UiXqUyOsEhEY2gWE)