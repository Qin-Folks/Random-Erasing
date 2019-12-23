def boo(func):
    def wrap():
        print('This is function: ', func.__name__)
        return func()
    return wrap

def foo():
    print('I am the bone of my sword')


c = boo @ foo

c()