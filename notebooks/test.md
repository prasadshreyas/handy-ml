# CS6140

Repositories

1. [Rushab](https://github.com/rushabhdharia/B555-MachineLearning) - 4 Years ago
2. [Anirudh](https://github.com/anirudhpillai16/Machine-Learning) - 6 Years ago
3. [Akhavani Northeastern](https://github.com/sa-akhavani/ml-cs6140/blob/master/hw1/main.pdf) - Spring 2020
4. [Piyush](https://github.com/piyushgoel997/MachineLearningAssignments/blob/master/Assignment1/main.pdf) - Spring 2020
5. https://github.com/PawPatel/Learning/
6. http://courses.enriqueareyan.com/files/computer-science/B555%20Machine%20Learning/Solutions/
7. https://github.com/SteinerSamuel/SchoolWork/tree/101e33ce3ebb2b668507ee50df0ed285cd955e65/Northeastern/CS6140%20-%20Machine%20Learning




---
# HW1

- [X] Q1 - Coin Toss
- [X] Q2 - Counter Example
- [ ] Q3 - Shopping
- [ ] Q4 - show that the density of X is a mixture of the densities of Y1 and Y
- [X] Q5 - Poisson MLE, MAP
- [X] Q6 - Normal Distribution MLE, MAP
- [ ] Q7 - EM Algorithm
  - [ ] a. 
  - [ ] b. 
  - [ ] c. Simulation
- [X] Q8 - High Dimensional Spaces
  - [ ] b. Need to Evaluate the ratio for different values
  - [ ] c. Simulation

---------------------

# 7 - EM Algorithm (Prob. 5 Assignment 2)


a) Derive update rules of an EM algorithm for estimating µ1, µ2, σ1 and σ2 based only on data set DY

b) Derive update rules of an EM algorithm for estimating α, β, µ1, µ2, σ1 and σ2 based on data set D

**Solution:**

If $Z_i$ are the hidden variables that indicate which of the two distributions the $i$th observation came from, then

$$ E_{Z} (log p(D,z|\theta^t)) $$

by using the formula 

$$\theta^{(t+1)} = \arg \max_{\theta} E_{Z} (log p(D,z|\theta^t))$$

$\theta$ is the set of parameters $\{ \alpha, \beta, \mu_1, \mu_2, \sigma_1, \sigma_2 \}$



------------------------------------------------------------------------
# 6

## what is the maximum a posteriori estimate of θ?

Maximum a posteriori is given by the following equation:

$$ \theta_{MAP} = \arg \max_{\theta}p(D| \theta) p(\theta) $$

where $p(\theta | D)$ is the posterior distribution of $\theta$ given the data $D$.





where $p(D|\theta)$ is the likelihood function and $p(\theta)$ is the prior distribution.



As we assume a normal distribution as the prior, the probability of a single observation $x_i$ given $\theta$ and $\sigma_0^2$ is:

$p(x_i|\theta, \sigma_0^2) = \frac{1}{\sqrt{2\pi\sigma_0^2}}exp(-\frac{(x_i-\theta)^2}{2\sigma_0^2})$

The likelihood of the data $D$ given $\theta$ is:

$$p(D|\theta) = \prod_{i=1}^n p(x_i|\theta, \sigma_0^2)$$



$= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_0^2}}exp(-\frac{(x_i-\theta )^2}{2\sigma_0^2})$

$= \frac{1}{\sqrt{2\pi\sigma_0^2}}exp(-\frac{1}{2\sigma_0^2}\sum_{i=1}^n (x_i-\theta)^2)$


The prior is given by:

$$p(θ) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(θ-\mu)^2}{2\sigma^2}}$$


Substituting the likelihood and prior into the MAP equation, we get:

$$θ_{MAP} = \argmax _θ {\frac{1}{\sqrt{2\pi\sigma_0^2}}exp(-\frac{1}{2\sigma_0^2}\sum_{i=1}^n (x_i-\theta)^2) \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(θ-\mu)^2}{2\sigma^2}}}$$

Taking the log of both sides, we get:

$$log(θ_{MAP}) = \log(\frac{1}{\sqrt{2\pi\sigma_0^2}}) + \log(exp(-\frac{1}{2\sigma_0^2}\sum_{i=1}^n (x_i-\theta)^2)) + \log(\frac{1}{\sqrt{2\pi\sigma^2}}) + \log(e^{-\frac{(θ-\mu)^2}{2\sigma^2}})$$

$$log(θ_{MAP}) = -\frac{1}{2}\log(2\pi\sigma_0^2) -\frac{1}{2\sigma_0^2}\sum_{i=1}^n (x_i-\theta)^2 -\frac{1}{2}\log(2\pi\sigma^2) -\frac{(θ-\mu)^2}{2\sigma^2}$$

Taking the derivative of the log of the MAP equation with respect to $\theta$ and setting it equal to zero, we get:

$$\frac{\partial log(θ_{MAP})}{\partial \theta} = -\frac{1}{\sigma_0^2}\sum_{i=1}^n (x_i-\theta) + \frac{(θ-\mu)}{\sigma^2} = 0$$

$$\sum_{i=1}^n (x_i-\theta) = \frac{\sigma_0^2}{\sigma^2}\theta - \frac{\sigma_0^2}{\sigma^2}\mu$$

$$\sum_{i=1}^n x_i - n\theta = \frac{\sigma_0^2}{\sigma^2}\theta - \frac{\sigma_0^2}{\sigma^2}\mu$$

$$\sum_{i=1}^n x_i = n\theta + \frac{\sigma_0^2}{\sigma^2}\mu$$

$$\theta = \frac{1}{n}\sum_{i=1}^n x_i + \frac{\sigma_0^2}{\sigma^2}\mu$$

---
Checking if the second derivative is negative, we get:

$$\frac{\partial^2 log(θ_{MAP})}{\partial \theta^2} = -\frac{n}{\sigma_0^2} - \frac{1}{\sigma^2}$$

Since, $\frac{n}{\sigma_0^2} > 0$ and $\frac{1}{\sigma^2} > 0$, the second derivative is negative and the maximum is a global maximum.

$$\frac{\partial^2 log(θ_{MAP})}{\partial \theta^2} < 0$$

## The Bayes estimate of θ is given by

The Bayes estimate of θ is given by:

$$θ_{B} = \int_{-\infty}^{\infty} \theta p(\theta|D) d\theta$$


By using proprotionality, we get:

$$θ_{B} \propto \int_{-\infty}^{\infty} \theta p(\theta|D) d\theta$$

Now eliminating the terms that do not depend on $\theta$, we get:

$$θ_{B} \propto e^{(\frac{1}{\sigma_0^2}\sum_{i=1}^n (x_i-\theta)^2 + \frac{(θ-\mu)^2}{\sigma^2})}$$


Simplying the power of $e$, we get:


$$θ_{B} \propto \frac{1}{\sigma_0^2}\sum_{i=1}^n (x_i-\theta)^2 +\frac{(θ-\mu)^2}{\sigma^2}$$

Expanding the square brackets, we get:

$$θ_{B} \propto \frac{1}{\sigma_0^2}\sum_{i=1}^n x_i^2 - 2\theta\sum_{i=1}^n x_i + n\theta^2 + \frac{θ^2 + \mu^2 - 2\theta\mu}{\sigma^2}$$


Assuming,

$$u_1 = \frac{\mu}{\sigma} + \frac{1}{\sigma_0^2}\sum_{i=1}^n x_i$$

$$\frac{1}{\sigma_1^2} = \frac{1}{\sigma_0^2}n + \frac{1}{\sigma^2}$$

Then the equation becomes:

$$θ_{B} \propto \frac{1}{\sigma_1^2}u_1^2 - 2\theta u_1 + \theta^2$$

$$\propto \theta^2 - 2\frac{u_1}{\sigma_1^2}\theta + \frac{u_1^2}{\sigma_1^2}$$

$$\propto \frac{1}{\sigma_1^2}(\theta - u_1)^2$$


The bayes estimate is just the mean of this distribution $\theta_{B} = u_1$ as the above equation is a normal distribution with mean $u_1$ and variance $\frac{1}{\sigma_1^2}$.


-------------------


# 8. High Dimensional Spaces

## a. Show that in a high dimensional space, most of the volume of a cube is concentrated in corners, which themselves become very long "spikes."

> Referred the [article](https://phys.libretexts.org/Bookshelves/Thermodynamics_and_Statistical_Mechanics/Book%3A_Statistical_Mechanics_(Styer)/10%3A_Appendices/10.04%3A_D-_Volume_of_a_Sphere_in_d_Dimensions) and the [article](https://en.wikipedia.org/wiki/Volume_of_an_n-ball) for the for the equation of volume of a hypersphere in n−dimensions of radius a

Let's say we have a cube of side length $2r$ in a $n$ dimensional space. The volume of the cube is given by:

$$V_c = (2r)^n$$


The volume of the sphere of radius $r$ is given by:

$$V_s = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)}r^n$$



where $\Gamma$ is the gamma function.

Then the ratio of the volume of the sphere to the volume of the cube is given by:

$$\frac{V_s}{V_c} = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)2^n}$$

Now, as $n$ increases, the ratio of the volume of the sphere to the volume of the cube approaches zero. This means that as $n$ increases, the volume of the cube is concentrated in the corners of the cube. This is shown by taking the limit as $n$ approaches infinity.

$$\lim_{n \to \infty} \frac{V_s}{V_c} = \lim_{n \to \infty} \frac{\pi^{n/2}}{\Gamma(n/2 + 1)2^n} = 0$$


To show that the corners become very long spikes, we compare the ratio of the distance between the corners of the cube to center of the cube and the distance between the center of the cube and the surface of the cube.

Calculating the distance from the center of the cube to the corner of the cube, we get:

$$d = \sqrt{(2r)^2 \times n} = 2r\sqrt{n}$$


We know that the distance from the center of the cube to the surface of the cube is $a$.

So the ratio of the distance between the corners of the cube to center of the cube and the distance between the center of the cube and the surface of the cube is given by:
$$\frac{d_{corner}}{d_{surface}} = \frac{2r\sqrt{n}}{r} = 2\sqrt{n}$$

As $n$ increases, the ratio of the distance between the corners of the cube to center of the cube and the distance between the center of the cube and the surface of the cube approaches infinity. This means that as $n$ increases, the corners become very long spikes.

-------------------

## Show that for points which are uniformly distributed inside a sphere in d dimensions where d is large, almost all of the points are concentrated in a thin shell close to the surface.

The volume of the sphere of radius $r$ from the previous question is given by:

$$V_s = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)}r^n$$

The volume of the sphere of radius $r_1 = r- \epsilon$ is given by:

$$V_{s_1} = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)}(r-\epsilon)^n$$

<!-- Then the ratio of the volume of the sphere of radius $r_1$ to the volume of the sphere of radius $r$ is given by:

$$\frac{V_{s_1}}{V_s} = \frac{\frac{\pi^{n/2}}{\Gamma(n/2 + 1)}(r-\epsilon)^n}{\frac{\pi^{n/2}}{\Gamma(n/2 + 1)}r^n}
$$ -->

 Then the ratio of the fraction of the volume of the sphere of radius $r_1$  and $r$ to the volume of the sphere of radius $r$ is given by:


$$ \frac{V_{s- s_1}}{V_s}  = \frac{\frac{\pi^{d/2}}{\Gamma(d/2 + 1)}r^d - \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}(r-\epsilon)^d}{\frac{\pi^{d/2}}{\Gamma(d/2 + 1)}r^d}$$

$$ = \frac{\frac{\pi^{d/2}}{\Gamma(d/2 + 1)}r^d}{\frac{\pi^{d/2}}{\Gamma(d/2 + 1)}r^d} - \frac{\frac{\pi^{d/2}}{\Gamma(d/2 + 1)}(r-\epsilon)^d}{\frac{\pi^{d/2}}{\Gamma(d/2 + 1)}r^d}$$

$$ = 1 - \frac{\pi^{d/2}}{\Gamma(d/2 + 1) r^d}(r-\epsilon)^d $$

Evaluate the above equation for $\epsilon \in \{ 0.01r,0.5r \}$ and $d \in \{1,2,3, 10, 100\}$. The results are shown in the table below.

| $\epsilon$ | $d$ | $\frac{V_{s- s_1}}{V_s}$ |
|:----------:|:---:|:-----------------------:|
| 0.01r      | 1   | 0.99                    |
| 0.01r      | 2   | 0.99                    |
| 0.01r      | 3   | 0.99                    |
| 0.01r      | 10  | 0.99                    |
| 0.01r      | 100 | 0.99                    |
| 0.5r       | 1   | 0.5                     |
| 0.5r       | 2   | 0.5                     |
| 0.5r       | 3   | 0.5                     |
| 0.5r       | 10  | 0.5                     |
| 0.5r       | 100 | 0.5                     |


As $\epsilon$ approaches zero, the ratio of the volume of the sphere of radius $r_1$ to the volume of the sphere of radius $r$ approaches one. This means that as $\epsilon$ approaches zero, the volume of the sphere of radius $r_1$ approaches the volume of the sphere of radius $r$. This means that as $\epsilon$ approaches zero, the points which are uniformly distributed inside a sphere in $d$ dimensions where $d$ is large are concentrated in a thin shell close to the surface.

----

# 4. 

Helpful links:

https://cedar.buffalo.edu/~srihari/CSE574/Chap9/Ch9.4-MixturesofBernoulli.pdf


















