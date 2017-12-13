class decorator_class(object):
    def __init__(self,arg):
        print('in decorator init, %s' % arg)
        self.arg = arg
    def __call__(self,function):
        print('in decorator call, %s' % self.arg)
        return function


decorator_instance = decorator_class('foo');

@decorator_instance
def function(*args, **kwargs):
    print('in function,%s %s' % (args,kwargs))


class replaceing_decorator_class(object):
    def __init__(self,arg):
        print('in decorator init, %s' % arg)
        self.arg = arg
    def __call__(self,function):
        print('in decorator call, %s' % self.arg)
        self.function = function
        return self.__wrapper__
    def __wrapper__(self,*args,**kwargs):
        print('in wrapper %s %s' % (args,kwargs))
        return self.function(args,kwargs)

new_deco_instance = replaceing_decorator_class('foo')


@new_deco_instance
def newFunction(*args, **kwargs):
    print('in newfunction,%s %s' % (args,kwargs))



class Foo(object):
    def __init__(self,data):
        self.data = data
        pass
    @classmethod
    def fromfile(cls,file):
        print('hahaha')
        pass

# with as 语法实际上是执行 类的__enter__方法，将返回值赋给 as后， 执行完代码后再执行 类的 __exit__方法
class exampleWith:
    def __enter__(self):
        print 'enter'
        return 'Foo'
    def __exit__(self,type,value,trace):
        print "type:", type
        print "value:", value
        print "trace:", trace
        print 'exit'
        pass

with exampleWith() as sample:
    print 'sample:', sample
    sample.do()
