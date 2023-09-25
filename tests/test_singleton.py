from unittest import TestCase
from yadeGrid.singleton import Singleton, SingletonMeta


class TestSingleton(TestCase):

    def test_singleton_decorator(self):
        @Singleton
        class MyClass:
            def __init__(self, name):
                self.name = name

        instance1 = MyClass("Alice")
        instance2 = MyClass("Bob")

        # The two instances should be identical
        self.assertIs(instance1, instance2)
        # The name should be "Alice" since the first instance created had that name
        self.assertEqual(instance1.name, "Alice")
        self.assertEqual(instance2.name, "Alice")

    def test_singleton_metaclass(self):
        class MyClass(metaclass=SingletonMeta):
            def __init__(self, name):
                self.name = name

        instance1 = MyClass("Charlie")
        instance2 = MyClass("David")

        # The two instances should be identical
        self.assertIs(instance1, instance2)
        # The name should be "Charlie" since the first instance created had that name
        self.assertEqual(instance1.name, "Charlie")
        self.assertEqual(instance2.name, "Charlie")
