# Introduction to JAX
### Arnur, Nithin

---

### What is JAX?

* JAX is a powerful Python library for numerical computing.
* JAX provides a high-level API for machine learning and scientific computing.

---

### Key Features of JAX

* NumPy compatibility.
* Automatic differentiation.
* GPU-accelerated computing.
* Functional programming (immutability, functions as first-class citizens).

---

### JAX Ecosystem

* Equinox: ML/neural networks.
* Haiku: ML/neural networks.
* Flax: ML/neural networks.
* jaxopt: optimization and gradient-based learning.
* Optax: optimization and gradient-based learning.
* Diffrax: numerical solving of ODEs and SDEs.
* ott-jax: optimal transport.

---

### Key transformations

* `jax.grad` for differentiation.
* `jax.vmap` for vectorization.
* `jax.jit` for compilation (speed-up).
* `jax.pmap` for sharding among multiple GPUs (not covered).

---

### First example

```python [1|1-18]
import jax.numpy as jnp

xs = jnp.linspace(-1.0, 1.0, 100)

def cheb_2(x):
    return jnp.cos(2 * jnp.arccos(x))

def cheb_3(x):
    return jnp.cos(3 * jnp.arccos(x))

def cheb_4(x):
    return jnp.cos(4 * jnp.arccos(x))

ys_2 = cheb_2(xs)
ys_3 = cheb_3(xs)
ys_4 = cheb_4(xs)
```

---

### Chebyshev polynomials
$$T_2(x) = 2x^2 - 1$$
$$T_3(x) = 4x^3 - 3x$$
$$T_4(x) = 8x^4 - 8x^2 + 1$$

----

![image](cheb.png)

---

### Immutability

`jax.numpy` arrays are immutable (like Python strings or tuples).

```python[1-10|5]
import jax.numpy as jnp

a = jnp.zeros((2, 2))

a[1, 1] = 22
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/narn/code/ml_constraints/jax_gpu_venv/lib/python3.11/site-packages/jax/_src/numpy/array_methods.py", line 270, in _unimplemented_setitem
    raise TypeError(msg.format(type(self)))
TypeError: '<class 'jaxlib.xla_extension.ArrayImpl'>' object does not support item assignment. JAX arrays are immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` or another .at[] method: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
```

---

### Immutability

If you really insist (but this is not in-place):

```python
import jax.numpy as jnp

a = jnp.zeros((2, 2))
b = a.at[1, :].set(22)
b
Array([[ 0.,  0.],
      [22., 22.]], dtype=float32)
```

---

### Differentiation: jax.grad

```python[1|2-5|7-9]
import jax

cheb_2_prime = jax.grad(cheb_2)
cheb_3_prime = jax.grad(cheb_3)
cheb_4_prime = jax.grad(cheb_4)

ys_2_prime_slow = []
for x in xs:
    ys_2_prime_slow.append(cheb_2_prime(x))
```

---

### Chebyshev polynomials: Derivatives
![image](cheb_prime.png)

---

### Vectorization: jax.vmap

* We want to apply `cheb_2_prime` to arrays, not just to scalars.
* This is called vectorization.
* Sometimes it happens automatically, but not in this case.
* We just need to use `jax.vmap`.

---

### Optional `vmap`

```python
ys_2 = cheb_2(xs)

v_cheb_2 = jax.vmap(cheb_2)
ys_2_vectorized = v_cheb_2(xs)
```

---

### Mandatory `vmap`

```python
# let's chain the transformations
v_cheb_2_prime = jax.vmap(jax.grad(cheb_2))
ys_2_prime = v_cheb_2_prime(xs)

# even shorter:
ys_3_prime = jax.vmap(jax.grad(cheb_3))(xs)
ys_4_prime = jax.vmap(jax.grad(cheb_4))(xs)
```

---

### JIT: compiling functions

* `jax.jit` does not change the signature.
* `jax.jit` makes functions faster.

```python
def f(x):
    z1 = x ** 2 + jnp.exp(x) + jnp.sin(x)
    z2 = 3 *(x ** 2 + jnp.exp(x) + jnp.sin(x)) + \
         2 * (jnp.sin(x) + jnp.cos(x**2 + jnp.exp(x)))
    return z1 + z2

x = jnp.linspace(0, 1, 1000000)
%timeit y = f(x).block_until_ready()
compiled_f = jax.jit(f)
# run once to compile
y_compiled = compiled_f(x)
# calls to compiled_f will be faster now
%timeit y_compiled = compiled_f(x).block_until_ready()
```

---

### GPU power

* By default, JAX places data on device (GPU).
    ```python
    print(xs.device())
    gpu:0
    ```
* It also reserves 75 % of GPU memory.
* We can change this, see
[documentation](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html).

---

## Detailed Look at Basic Concepts

---

### Vectorization: jax.vmap

* $f \colon \mathbb{R} \to \mathbb{R}$.
* Vectorization of $f$ : $f^v \colon \mathbb{R}^n \to \mathbb{R}^n$,
$(x_1, \dots, x_n) \mapsto (f(x_1), \dots, f(x_n))$.
* NumPy does it implicitly, it is convenient.
* JAX supports this because of compatibility of NumPy, but not everywhere.
* A better option is *explicit* vectorization: `jax.vmap`.

---

### vmap

$f \colon \mathbb{R}^n \to \mathbb{R}^m$.

* `jax.vmap` adds one more axis (dimension) to input and output.
* `jax.vmap(f)` applies $f$ to a matrix row-wise: $\mathbb{R}^{k \times n} \to \mathbb{R}^{k \times m}$
* Standard *feature matrix* in ML: rows correspond to samples, columns are features.

---

### vmap

$f \colon \mathbb{R}^n \to \mathbb{R}^m$, think of $f(x)$ as row vectors.

`jax.vmap(f)`:

$$\begin{pmatrix}
x^{(1)}_1 & x^{(1)}_2 \dots & x^{(1)}_n \\\\
\dots & & \\\\
x^{(k)}_1 & x^{(k)}_2 \dots & x^{(k)}_n
\end{pmatrix} \mapsto
\begin{pmatrix}
f(x^{(1)}) \\\\
\dots  \\\\
f(x^{(k)})
\end{pmatrix}
$$


---

### vmap

$f \colon \mathbb{R}^n \to \mathbb{R}^m$.

* If we want to apply `f` column-wise, we can do it with `in_axes=1`.
* `jax.vmap(f, in_axes=1)` maps $\mathbb{R}^{n \times k} \to \mathbb{R}^{k \times m}$
* The result is still accumulated along axis 0 (as rows).

---

### vmap

$f \colon \mathbb{R}^n \to \mathbb{R}^m$,

`jax.vmap(f, in_axes=1)`:
$x$ is a column vector, $f(x)$ is a row vector.
$$\begin{pmatrix}
x^{(1)}_1 & x^{(2)}_1 \dots & x^{(k)}_1 \\\\
\dots & & \\\\
x^{(1)}_n & x^{(2)}_n \dots & x^{(k)}_n
\end{pmatrix} \mapsto
\begin{pmatrix}
f(x^{(1)}) \\\\
\dots  \\\\
f(x^{(k)})
\end{pmatrix}
$$


---

### vmap

$f \colon \mathbb{R}^n \to \mathbb{R}^m$.

* We can accumulate the results along another axis (as columns) with `out_axes`.
* `jax.vmap(f, out_axes=1)` maps $\mathbb{R}^{k \times n} \to \mathbb{R}^{m \times k}$

---

### vmap

$f \colon \mathbb{R}^n \to \mathbb{R}^m$.

* Of course, we can use both `in_axes` and `out_axes`:
* `jax.vmap(f, in_axes=1, out_axes=1)` maps $\mathbb{R}^{n \times k} \to \mathbb{R}^{m \times k}$.

---

### vmap

* If `in_axes` is an `int`, it adds an axis to **all** input arguments.
* What if we don't want to do this?
* What if we want to add an extra axis at different positions for different
arguments?
* `in_axes` can be a tuple with `len` equal to the number of arguments.


---

### vmap

Example: kinetic energy along trajectory in the 3-body problem.

```python
def kinetic_energy(p, m1, m2, m3):
    kin_1 = m1 * jnp.sum(p[9:12] ** 2)
    kin_2 = m2 * jnp.sum(p[12:15] ** 2)
    kin_3 = m3 * jnp.sum(p[15:18] ** 2)
    return (kin_1 + kin_2 + kin_3) / 2

key = jrandom.PRNGKey(42)
traj = jrandom.uniform(key, (100, 18))

m1 = m2 = m3 = 1.0
v_energy = jax.vmap(kinetic_energy,
        in_axes=(0, None, None, None))
ys = v_energy(traj, m1, m2, m3)
print(ys.shape)
(100,)
```
---

### vmap

* In one call to `vmap` we can add 0 or 1 axis to each argument
* If we need to add more axes, just apply `vmap` again.

---

## Differentiation: Details

---

### Differentiation: Mathematics

* $$\mbox{if }f \colon \mathbb{R} \to \mathbb{R},\mbox{ then } \nabla(f)=\frac{df}{dx} \colon \mathbb{R} \to \mathbb{R}.$$
* $\nabla$ takes a function and returns a function.
* $$\nabla \colon C^\infty(\mathbb{R}) \to C^\infty(\mathbb{R}).$$

---

### Differentiation: jax.grad

* Functional programming: functions as first-class citizens.
* `jax.grad` is a *transformation*, it takes functions and returns functions.
* `jax.grad` *is* $\nabla$.
* NB: `jax.grad` is defined only for scalar-valued functions, not
vector-valued.

---

JAX can see only through JAX functions.

```python[6|9]
import jax
import jax.numpy as jnp
import numpy as np

def cheb_2(x):
    return np.cos(2 * jnp.arccos(x))

cheb_2_prime = jax.grad(cheb_2)
cheb_2_prime(1.0)
```

```sh
    cheb_2_prime(1.0)
 File "...", line 6, in cheb_2
    return np.cos(2 * jnp.arccos(x))
           ^^^^^^^^^^^^^^^^^^^^^^^^^
jax.errors.TracerArrayConversionError: The numpy.ndarray
conversion method __array__() was called on traced array
with shape float32[].
```

---

### Differentiation: Details

By default, `jax.grad` takes the derivative w.r.t. first argument.
If $f \colon \mathbb{R}^n \to \mathbb{R}$ is implemented as
* ```python
def f(x):
    # x is an n-dimensional array
```
then `jax.grad` is exactly $\nabla$.
*
```python
def f(x_1, ..., x_n):
    # x_1, ..., x_n are reals
```
then `jax.grad` is $\frac{\partial f}{\partial x_1}$.

---

### Differentiation: Details

```python
import jax

def f(x, y):
    return x ** 2 + y ** 3

f_x = jax.grad(f)
print(f_x(2.0, 3.0))

4.0
```

---

### Differentiation: Details

What if the first argument is not scalar?
The output still must be a scalar.


```python
def g(x, y):
    return jnp.sum(x ** 2 + y ** 3)

x = jnp.ones((2, 2))
y = jnp.ones((2, 2))

g_x = jax.grad(g)

print(g_x(x, y))

[[2. 2.]
 [2. 2.]]
```

---

### Differentiation: Details

What if we want the derivatives w.r.t. more arguments?

```python
def h(x, y, z):
    return x ** 2 + y ** 3 + z ** 4

h_y = jax.grad(h, argnums=1)
h_x_and_h_z = jax.grad(h, argnums=(0,2))

print(h_y(2, 3, 4))
27.0

print(h_x_and_h_z(2, 3, 4))
(Array(4., dtype=float32, weak_type=True),
 Array(256., dtype=float32, weak_type=True))
```

---

### Speed: JIT

* `jax.jit` compiles a function
* `jit` does not change the signature

Big restrictions:

1. Shapes of inputs must be fixed
2. No conditioning on values


---

### JIT: Pure functions

* All `jit`-ted functions are pure.
* Side effects of the original functions are usually lost.
* They can occur once during *tracing*, but you should never rely on that.

---

### Example: no JIT

```python
#!python3

import jax.numpy as jnp

global_list = []

def log2_with_print(x):
    global_list.append(x)
    print("printed x:", x)
    return jnp.log(x) / jnp.log(2.0)

log2_with_print(2.0)
log2_with_print(4.0)
log2_with_print(6.0)
print(global_list)
```

```sh
printed x: 2.0
printed x: 4.0
printed x: 6.0
[2.0, 4.0, 6.0]
```

---

### Example: JIT tracing

```python
#!python3

import jax
import jax.numpy as jnp


def log2_with_print(x):
    print("printed x:", x)
    return jnp.log(x) / jnp.log(2.0)

jlog2_with_print = jax.jit(log2_with_print)

jlog2_with_print(2.0)
jlog2_with_print(4.0)
jlog2_with_print(6.0)
```

```sh
printed x: Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJAXprTrace(level=1/0)>
```

---


### Example: JIT tracing

```python
#!python3

import jax
import jax.numpy as jnp


def log2_with_print(x):
    print("printed x:", x)
    return jnp.log(x) / jnp.log(2.0)

jlog2_with_print = jax.jit(log2_with_print)

jlog2_with_print(2.0)
jlog2_with_print(4.0)
jlog2_with_print(6.0)
jlog2_with_print(jnp.array([16.0, 32.0]))
```

```sh
printed x: Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJAXprTrace(level=1/0)>
printed x: Traced<ShapedArray(float32[2])>with<DynamicJAXprTrace(level=1/0)>
```

---

### JIT tracing

* `Traced` is *not* convertible to numbers: there is no way to access concrete values.
* Think of it as mathematical variable: if input is $x$ and $y_1 = x^2$,
$y_2 = 1 / x$, and we `return` $y_1 - y_2$, then tracing allows us to realize
that we return $x^2 - 1/ x$.
* We do not need specific values to figure that, better to stick to algebra and
letters.
* That is, roughly speaking, what `jax.jit` does, plus optimizations of the
final expression.

---

### No conditioning on values

```python[1-9|7,9]
def foo(x, y):
    if y == 1:
        return jnp.abs(x)
    elif y == 2:
        return jnp.sum(x ** 2)

foo = jax.jit(foo)

foo(1.0, 1)
```

```sh
Traceback (most recent call last):
  File "/home/narn/code/ml_constraints/jax_tutorial/jit_example_3.py", line 14, in <module>
    foo(1.0, 1)
  File "/home/narn/code/ml_constraints/jax_tutorial/jit_example_3.py", line 7, in foo
    if y == 1:
       ^^^^^^
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape bool[].
The problem arose with the `bool` function.
The error occurred while tracing the function foo at /home/narn/code/ml_constraints/jax_tutorial/jit_example_3.py:6 for jit. This concrete value was not available in Python because it depends on the value of the argument y.
```

---

### Static arguments

We can tell JAX to treat this argument differently.

```python[1-10|7]
def foo(x, y):
    if y == 1:
        return jnp.abs(x)
    elif y == 2:
        return jnp.sum(x ** 2)

foo = jax.jit(foo, static_argnums=1)

foo(1.0, 1)
```
---

### Static arguments

There is a price: JAX has to recompile for each new value of `y`.

```python[0-15|1,6|15,16]
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=1)
def foo(x, y):
    print(x, y)
    if y == 1:
        return jnp.abs(x)
    elif y == 2:
        return jnp.sum(x ** 2)

for y in range(1, 5):
    foo(1.0, y)
```

---

### Static arguments

```python
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=1)
def foo(x, y):
    print(x, y)
    if y == 1:
        return jnp.abs(x)
    elif y == 2:
        return jnp.sum(x ** 2)

for y in range(1, 5):
    foo(1.0, y)
```

```
Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJAXprTrace(level=1/0)> 1
Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJAXprTrace(level=1/0)> 2
Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJAXprTrace(level=1/0)> 3
Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJAXprTrace(level=1/0)> 4
```

---

### Static arguments

* If you expect a static argument to have a few distinct values, `jax.jit` is
fine.
* Another use case: class methods.
* If you expect it to have many different values, `jax.jit` will slow you down.
* You can also use `static_argnames`.

---

### How it works?

* `jax` maintains `dict` with compiled versions of functions.
* Keys: *types* and *shapes* of non-static arguments, *values* of static ones.
* If there is a compiled function for a key, call it.
* Otherwise, trace and add one more entry to the dictionary.

---

### How it works?

```python[1-9|3-5|3-5,7|3-5,8|3-5,9]
f = jax.jit(f)

x = jnp.ones((2, 2))
y = 3 * jnp.ones((2, 2))
z = jnp.ones((3, 3, 3))

f(x)
f(y)
f(z)
```

---

### Functional programming

Functions (mappings, operators, morphisms, etc) in mathematics vs functions in programming:

* Functions are pure.
* Often take other functions as arguments (**higher-order**).
    * Fourier transform: takes $f \colon \mathbb{R} \to \mathbb{R}$, returns $f \colon \mathbb{R} \to \mathbb{R}$.
    * Covariant derivative: takes a vector and vector field (which, in turn, is
            a function $M \to TM$), returns a vector.

---

### Functional programming: Pure functions

* Deterministic: same arguments always produce same inputs
* No side effects, just the returned values.
* What about `sin`? What about `time.time`? What about `print`?
* What about 'given file name, return the number of lines in the file'?

---

### Functional programming: Pure functions

Side effects examples:

* Changing global variables.
* Changing hidden state.
* Printing.
* Sending data to a remote server.
* Reading file.

---

### Functional programming

Functions are *first-class citizens*:

* We store them in variables.
* Give them to functions and get back new functions.

---

### Functional programming: Pure functions and state

* What if we need state?
* Pass it as an argument.
* Example: PRNG (pseudo-random number generators).

---

### Random numbers in JAX

Randomness is needed in:

* Initializing weights of NNs.
* Train-test split.
* Shuffling training dataset for batches.

---

### Random numbers in JAX

What about reproducibility? Traditional way:
```python
seed = 42
random.seed(42)
np.random.seed(42)
torch.random.seed(42)
...
```

---

### Traditional PRNG

* State of the pseudo-random number generator is hidden.
* The state is modified under the hood each time a new random number is generated.


```python
_global_random_state = ...

def set_seed(s):
    _global_random_state = # concatenate s to get n digits

def get_random():
    result = middle_n_digits(global_random_state ** 2)
    _global_random_state = result
    return result
```

---

### Sequential Consistency

```python
np.random.seed(0)
print("individually:",
        np.stack([np.random.uniform() for _ in range(3)]))

np.random.seed(0)
print("all at once: ", np.random.uniform(size=3))
```
```sh
individually: [0.5488135  0.71518937 0.60276338]
all at once:  [0.5488135  0.71518937 0.60276338]
```

---

### Random numbers in JAX

* What if the data live on multiple GPUs?
* What if our code is run in parallel?
* We still want reproducibility.
* FP ideology of pure functions is a good fit.

---

### Random numbers in JAX

JAX way:

* Create *keys*.
* Use each key only once.
* Create key from seed once, then *split*.
* Functions taking keys are deterministic.

```python
import jax.random as jrandom

seed = 42
key = jrandom.PRNGKey(seed)

next_key, subkey = jrandom.split(key)
del key
xs = jrandom.uniform(subkey, (3,))
```

---

### Random numbers in JAX

Idiomatic way:

```python
import jax.random as jrandom

seed = 42
key = jrandom.PRNGKey(seed)

key, subkey = jrandom.split(key)
xs = jrandom.uniform(subkey, (3,))

# next time we need something random
key, subkey = jrandom.split(key)
ys = jrandom.normal(subkey, (10,10))
```

---

### Random numbers in JAX

Idiomatic way:

```python
import jax.random as jrandom

seed = 42
key = jrandom.PRNGKey(seed)

# we need 10 random normal matrices and 1 random vector
key, *subkeys = jrandom.split(key, num=12)
xs = jrandom.uniform(subkeys[0], (3,))

rand_matrices = [ jrandom.normal(subkeys[i], (20, 20))
                  for i in range(1, 11) ]
print(len(rand_matrices))
10
```

---

### No sequential equivalence guarantee

```python
key = random.PRNGKey(42)
subkeys = random.split(key, 3)
sequence = np.stack([random.normal(subkey) for subkey in subkeys])
print("individually:", sequence)

key = random.PRNGKey(42)
print("all at once: ", random.normal(key, shape=(3,)))
```

```sh
individually: [-0.04838839  0.10796146 -1.2226542 ]
all at once:  [ 0.18693541 -1.2806507  -1.5593133 ]
```

---

### PyTrees

* We want gradients on the parameters of NNs.
* `jax.grad` takes the derivatives w.r.t. first argument.
* How complicated can it be?
* Can we encode the whole model there?

---

### Differentiation: PyTrees

* Yes, we can pass the whole NN as an argument!
* ...if we pack it in a pytree.

----

> A pytree is a container of leaf elements and/or more pytrees. Containers include lists, tuples, and dicts.
> A leaf element is anything that’s not a pytree, e.g. an array. In other words, a pytree is just a possibly-nested standard or
> user-registered Python container.
> If nested, note that the container types do not need to match. A single “leaf”, i.e. a non-container object, is also considered a pytree.

---

### Examples

```
import jax
import jax.numpy as jnp

example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

# Let's see how many leaves they have:
for pytree in example_trees:
    leaves = jax.tree_util.tree_leaves(pytree)
    print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")
```

----

```sh
[1, 'a', <object object at 0x11b952310>]      has 3 leaves: [1, 'a', <object object at 0x11b952310>]
(1, (2, 3), ())                               has 3 leaves: [1, 2, 3]
[1, {'k1': 2, 'k2': (3, 4)}, 5]               has 5 leaves: [1, 2, 3, 4, 5]
{'a': 2, 'b': (2, 3)}                         has 3 leaves: [2, 2, 3]
Array([1, 2, 3], dtype=int32)                 has 1 leaves: [Array([1, 2, 3], dtype=int32)]

```

---

### PyTree in action

```python
import jax.numpy as jnp
import jax
import jax.random as jrandom
from matplotlib import pyplot as plt

true_params = (-0.5, 2.3, 1.0)

@jax.jit
def predict(params, x):
    k_1, k_2, b = params
    k = jax.lax.cond(x < 0, lambda _: k_1, lambda _: k_2, x)
    return k * x + b

xs = jnp.linspace(-1, 1, 100)
vpredict = jax.vmap(predict, in_axes=(None, 0))
ys = vpredict(true_params, xs)
```

----

```python
noise = jrandom.uniform(jrandom.PRNGKey(42), ys.shape, minval=-0.2,
        maxval=0.2)

noisy_ys = ys + noise

@jax.jit
def loss(params, xs, ys):
    pred_ys = vpredict(params, xs)
    return jnp.sum((ys - pred_ys) ** 2)

k_1, k_2, b = jrandom.uniform(jrandom.PRNGKey(seed), (3,))

params = [k_1, k_2, b]

n_steps = 30
lr = 0.005
```

----

```python
loss_and_grad = jax.value_and_grad(loss)

for step in range(n_steps):
    curr_loss, params_grad = loss_and_grad(params, xs, noisy_ys)

    print(f"{step = }, {params_grad = }, {params = }")

    for j in range(len(params)):
        params[j] -= lr * params_grad[j]

    if step % 4 == 0:
        plt.plot(xs, vpredict(params, xs), color="red")
        plt.plot(xs, noisy_ys, color="green")
        plt.show()
```

---

### Step 1:
![image](regr_kink_0.png)

----

### Step 5:
![image](regr_kink_1.png)

----

### Step 15:
![image](regr_kink_2.png)


----

### Step 30:
![image](regr_kink_3.png)

---

### Sharp Bits 1:

Out-of-bounds index does **not** cause an error (welcome to the painful C++
        world with UB).

```python
>>> import jax.numpy as jnp
>>> import numpy as np
>>> a_jnp = jnp.arange(10)
>>> a_np = np.arange(10)
>>> a_jnp[10] # should be an error, but...
Array(9, dtype=int32)
>>> a_jnp[1000] # even this:
Array(9, dtype=int32)
>>> a_np[10]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 10 is out of bounds for axis 0 with size 10
```

---

### Sharp Bits 2: NaNs

```python
def cheb_2(x):
    return jnp.cos(2 * jnp.arccos(x))

xs = jnp.linspace(-1.0, 1.0, 4)

cheb_2_prime = jax.grad(cheb_2)
print(jax.vmap(cheb_2_prime)(xs))
[       inf -1.3333331  1.3333337        nan]
```

---

### Sharp Bits 2: NaNs

We must tell JAX that we want it to raise an exception.


```python
jax.config.update("jax_debug_nans", True)

def cheb_2(x):
    return jnp.cos(2 * jnp.arccos(x))

xs = jnp.linspace(-1.0, 1.0, 4)

cheb_2_prime = jax.grad(cheb_2)
print(jax.vmap(cheb_2_prime)(xs))
...
    print(jax.vmap(cheb_2_prime)(xs))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
FloatingPointError: invalid value (nan) encountered in jit(mul)
```

---

### Sharp Bits 3: 32 bits by default

If you want to use `double` (64 bit precision):

1. Set the environment variable `JAX_ENABLE_X64=True`.
2. **On startup**:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

You can also use `absl` configuration, see
   [docs](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

---


### Sharp bit in profiling: block_until_ready

* By default, JAX is asynchronous: calls return after computations are
scheduled.
* To measure running time we need to use `block_until_ready` method.

**Wrong**:
```python
import time

x = jnp.random.uniform(jrandom.PRNGKey(42), (10000, 10000))

start = time.time()
f(x)
elapsed = time.time() - start
```

---

### Profiling: block_until_ready

* By default, JAX is asynchronous: calls return after computations are
scheduled.
* To measure running time we need to use `block_until_ready` method.

**Right**:
```python[1-8|6]
import time

x = jnp.random.uniform(jrandom.PRNGKey(42), (10000, 10000))

start = time.time()
f(x).block_until_ready()
elapsed = time.time() - start
```

---
### Implicit Function Theorem

* An extremely powerful theorem when applied to differentiable programming.
* An alternative to backpropagation.
* But why? A complex trade-off between memory and compute.

---
### Fixed point iteration

* Suppose we want to perform a fixed point iteration:
* $z = \tanh(Wz + x)$ (Very very simple!)
* Then define $f(x,z) = z - \tanh(Wz + x)$.
* Find $z^*$ such that $f(x, z^*) = 0$.
* How to find $\frac{\partial z^*}{ \partial x}$?

---
### IFT Statement

* Let $f: \mathbb{R}^p \times \mathbb{R}^n \rightarrow \mathbb{R}^n$ and $a_0 \in \mathbb{R}^p$, $z_0 \in \mathbb{R}^n$ be such that:

1. $f(a_0, z_0) = 0$ and
2. $f$ is continuously differentiable with non-singular Jacobian $\partial_1 f(a_0, z_0) \in \mathbb{R}^{n\times n}$

* Then there exist open sets $S_{a_0} \subset \mathbb{R}^p$ and $S_{z_0} \subset \mathbb{R}^n$ containing $a_0$ and $z_0$, respectively, and a unique continuous function $z^*: S_{a_0} \rightarrow S_{z_0}$ such that:

1. $z_0 = z^*(a_0)$
2. $f(a, z^*(a)) = 0 \forall a \in S_{a_0}$, and
3. $z^*$ is differentiable on $S_{a_0}$.

---
### Fixed Point Iteration (Again)

* We know via iteration $f(a, z^*(a)) = 0$.
* Then: $\partial_0 f(a_0, z_0) + \partial_1 f(a_0, z_0)\partial z^*(a_0) = 0$
* $\rightarrow$ $\partial z^*(a) = - \left[ \partial_1 f(a_0, z_0)\right]^{-1} \partial _0 f(a_0, z_0)$
---

### Even ChatGPT knows about JAX:

JAX Workflow

1. Define a computation using JAX arrays and functions.
2. Transform the computation with `jit` and `vmap` to optimize and parallelize the code.
3. Execute the transformed computation on CPUs, GPUs, or TPUs.
4. Access gradients with JAX's automatic differentiation.


* There are lots of things that we didn't even mention, see the docs.

# Happy JAX-ing

---

# Assorted Slides

---

### Installation

* We are not talking about installation.
* See [this page](https://github.com/google/jax#installation) for detailed
instructions.
* In general, [JAX documentation](https://jax.readthedocs.io) is extremely well-written.

---

### Differentiation: Auxilliary Info

`has_aux=True` allows us to return one more piece of information from the
gradient.

```python[1-15|9]
const_g = 9.81

def lagrangian(p, m):
    # p: phase point (x, y, z, v_x, v_y, v_z)
    kinetic = m * jnp.sum(p[3:6] ** 2) / 2
    potential = m * const_g * p[2]
    return kinetic - potential, (kinetic, potential)

grad_lagr = jax.grad(lagrangian, has_aux=True)

p = jnp.array([2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
lagr_gradient, (kinetic, potential) = grad_lagr(p, 1.0)
print(lagr_gradient, (kinetic, potential))
[ 0.    0.   -9.81  1.    2.    3.  ]
(Array(7., dtype=float32), Array(39.24, dtype=float32))
```

---

### Differentiation: Auxilliary Info

$$ \mathcal{L} = T - U $$
We computed $\nabla \mathcal{L}$, $T$ and $U$ in a couple of lines.

```python
def lagrangian(p, m):
    ...
    return kinetic - potential, (kinetic, potential)

grad_lagr = jax.grad(lagrangian, has_aux=True)

lagr_gradient, (kinetic, potential) = grad_lagr(p, 1.0)
```

---


### Differentiation: value_and_grad and has_aux

```python[1-15|9|9,11]
def lagrangian(p, m):
    # p: phase point (x, y, z, v_x, v_y, v_z)
    kinetic = m * jnp.sum(x[3:6] ** 2) / 2
    potential = m * const_g * x[2]
    return kinetic - potential, (kinetic, potential)

p = jnp.array([2.0, 3.0, 4.0, 1.0, 2.0, 3.0])

llp = jax.value_and_grad(lagrangian, has_aux=True)

(lagr, (kin, pot)), lagr_gradient = llp(p, 1.0)
print(lagr, kin, pot, lagr_gradient)
-32.24 7.0 39.24 [ 0.    0.   -9.81  1.    2.    3.  ]
```

---

### Differentiation: Auxilliary Info

$$ \mathcal{L} = T - U $$

* $\mathcal{L}$  (`value_and_grad`)
* $T$ and $U$ (`has_aux=True`, second component of `value`)
* $\nabla \mathcal{L}$  (`value_and_grad`)

```python[1-15|9|9,11]
def lagrangian(p, m):
    ...
    return kinetic - potential, (kinetic, potential)

llp = jax.value_and_grad(lagrangian, has_aux=True)

(lagr, (kin, pot)), lagr_gradient = llp(p, 1.0)
```

---

### Preallocation

 We can change this behavior.

* To suppress preallocation:
    ```sh
    $ export XLA_PYTHON_CLIENT_PREALLOCATE=false
    $ python
    >>> import jax
    >>> import jax.numpy as jnp
    ```
    or, if you want to do it from python,
    ```python
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax
    ```

---

### Preallocation

* To change the percentage of pre-allocated memory to 10 %:
    ```sh
    $ export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
    $ python
    >>> import jax
    >>> import jax.numpy as jnp
    ```
    or, if you want to do it from python,
    ```python
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".1"

    import jax
    ```
---

### vmap

For example, to apply function to a matrix entry-wise.

```python
def pdf(x, params):
    mu, sigma = params
    a = ((x - mu) / sigma) ** 2
    b = sigma * jnp.sqrt(2 * jnp.pi)
    return jnp.exp(-a/2) / b

params = jnp.array([0.2, 2.0])

key = jrandom.PRNGKey(42)
key, subkey_1, subkey_2 = jrandom.split(key, 2)
xs_vec = jrandom.uniform(subkey_1, (20,))
xs_matr = jrandom.uniform(subkey_2, (30, 30))

pdf_vec = jax.vmap(pdf, in_axes=(0, None))
pdf_matr = jax.vmap(pdf_vec, in_axes=(0, None))

ys_vec = pdf_vec(xs_vec, params)
ys_matr = pdf_matr(xs_matr, params)
```

---

### vmap

```python
def cond_plus_cond_sq(m):
    c =jnp.linalg.cond(m)
    return c + c ** 2

m_1 = jnp.array([[1.0, 0.0], [0.0, 100.0]])
m_2 = jnp.array([[1.0,  1.0], [-1.0, 1.0]])
m = jnp.array((m_1, m_2))

print(jax.vmap(cond_plus_cond_sq)(m))

[1.01e+04 2.00e+00]
```
---

### vmap

```python
def cond_plus_cond_sq(m):
    c =jnp.linalg.cond(m)
    return c + c ** 2

m_1 = jnp.array([[1.0, 0.0], [0.0, 100.0]])
m_2 = jnp.array([[1.0,  1.0], [-1.0, 1.0]])
m = jnp.array((m_1, m_2))

print(jax.vmap(cond_plus_cond_sq, in_axes=2)(m))
[9.4721365e+00 1.0102011e+04]

for i in range(m.shape[2]):
    x = m[:, :, i].reshape(2, 2)
    c = jnp.linalg.cond(x)
    print(x, c, c + c ** 2)

[[ 1.  0.]
 [ 1. -1.]] 2.6180341 9.4721365
[[  0. 100.]
 [  1.   1.]] 100.01 10102.011
```

---

### vmap

```python
foo = jax.vmap(cond_plus_cond_sq, in_axes=2)(m)

# equivalent to:

def foo(m):
    result = []
    for i in range(m.shape[2]):
        x = m[:, :, i].reshape(m.shape[0], m.shape[1])
        result.append(cond_plus_cond_sq(x))
    return jnp.array(result)
```

---


### vmap

We can arrange the result along different axis, too.

```python
foo = jax.vmap(cond_plus_cond_sq, in_axes=2, out_axes=1)(m)

# equivalent to:

def foo(m):
    result = []
    for i in range(m.shape[2]):
        x = m[:, :, i].reshape(m.shape[0], m.shape[1])
        result.append(cond_plus_cond_sq(x))
    return jnp.array(result).T
```


---

### Differentiation: value_and_grad

```python[1-15|9]
const_g = 9.81

def lagrangian(p, m):
    # p: phase point (x, y, z, v_x, v_y, v_z)
    kinetic = m * jnp.sum(p[3:6] ** 2) / 2
    potential = m * const_g * p[2]
    return kinetic - potential

lagr_and_grad_lagr = jax.value_and_grad(lagrangian)

p = jnp.array([2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
lagr, lagr_gradient = lagr_and_grad_lagr(p, 1.0)
```

---

### Differentiation: value_and_grad

$$\mathcal{L} = T - U$$

We computed $\mathcal{L}$ and $\nabla \mathcal{L}$:

```python
def lagrangian(p, m):
    ...
    return kinetic - potential

lagr_and_grad_lagr = jax.value_and_grad(lagrangian)
```

---


### Profiling: Compilation Takes Time

* `jit`-ting takes time.
* To have a fair comparison, run the `jit`-ted function first
on arguments with same shape.

---

### Profiling: Compilation Takes Time

```python
import time
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0,
                               x,
                               alpha * jnp.exp(x) - alpha)

x = jnp.arange(100000)
```

----

```python[0-15|7|9-11]
start = time.time()
selu(x).block_until_ready()
elapsed_no_jit = time.time() - start

selu_jit = jax.jit(selu)

selu_jit(x).block_until_ready()

start = time.time()
selu_jit(x).block_until_ready()
elapsed_jit = time.time() - start

print(f"{elapsed_no_jit = :0.6}\n   {elapsed_jit = }")
elapsed_no_jit = 0.412574
   elapsed_jit = 0.000113
```

---

### Jit: remarks

* `jit` can make computations much more efficient.
* `jit`-ting a `vmap`-ped function makes sense.
* All side effects are lost.
* There are no guarantees about side effects during tracing.
* Using `jnp.where` or `jax.lax.cond` may be an alternative to
`static_argnums`.

---
