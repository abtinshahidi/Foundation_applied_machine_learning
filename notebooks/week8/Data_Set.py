from utilities import _read_data_set, remove_all, unique, mean_boolean_error 




class Data_Set:
    """
    Defining a _general_ data set class for machine learning.

    These are the following fields:

    >> data = Data_set

    data.examples:

           list of examples. Each one is a list contains of attribute values.


    data.attributes:

           list of integers to index into an example, so example[attribute]
           gives a value.


    data.attribute_names:

           list of names for corresponding attributes.


    data.target_attribute:

           The target attribute for the learning algorithm.
           (Default = last attribute)


    data.input_attributes:

           The list of attributes without the target.



    data.values:

           It is a list of lists in which each sublist is the
           set of possible values for the corresponding attribute.
           If initially None, it is computed from the known examples
           by self.setproblem. If not None, bad value raises ValueError.


    data.distance_measure:

           A measure  of  distance  function which takes two examples
           and returns a nonnegative number. It should be a symmetric
           function.
           (Defaults = mean_boolean_error) : can handle any field types.


    data.file_info:

           This should be a tuple that contains:
        (file_address, number of rows to skip, separator)



    data.name:

           This is for naming the data set.



    data.source:

            URL or explaination to the dataset main source


    data.excluded_attributes:

            List of attribute indexes to exclude from data.input_attributes.
            (indexes or names of the attributes)

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs."""




    def __init__(self, examples=None, attributes=None,  attribute_names=None,
                 target_attribute = -1,  input_attributes=None,  values=None,
                 distance_measure = mean_boolean_error, name='',  source='',
                 excluded_attributes=(), file_info=None):

        """
        Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .setproblem().

        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """

        self.file_info = file_info
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance_measure
        self.check_values_flag = bool(values)

        # Initialize examples from a list
        if examples is not None:
            self.examples = examples
        elif file_info is None:
            raise ValueError("No Examples! and No Address!")
        else:
            self.examples = _read_data_set(file_info[0], file_info[1], file_info[2])

        # Attributes are the index of examples. can be overwrite
        if self.examples is not None and attributes is None:
            attributes = list(range(len(self.examples[0])))

        self.attributes = attributes

        # Initialize attribute_names from string, list, or to default
        if isinstance(attribute_names, str):
            self.attribute_names = attribute_names.split()
        else:
            self.attribute_names = attribute_names or attributes

        # set the definitions needed for the problem
        self.set_problem(target_attribute, input_attributes=input_attributes,
                         excluded_attributes=excluded_attributes)



    def get_attribute_num(self, attribute):
        if isinstance(attribute, str):
            return self.attribute_names.index(attribute)
        else:
            return attribute




    def set_problem(self, target_attribute, input_attributes=None, excluded_attributes=()):
        """
        By doing this we set the target, inputs and excluded attributes.

        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attrname.
        Also computes the list of possible values, if that wasn't done yet."""

        self.target_attribute = self.get_attribute_num(target_attribute)

        exclude = [self.get_attribute_num(excluded) for excluded in excluded_attributes]

        if input_attributes:
            self.input_attributes = remove_all(self.target_attribute, input_attributes)
        else:
            inputs = []
            for a in self.attributes:
                if a != self.target_attribute and a not in exclude:
                    inputs.append(a)
            self.input_attributes = inputs

        if not self.values:
            self.update_values()
        self.sanity_check()


    def sanity_check(self):
        """Sanity check on the fields."""

        assert len(self.attribute_names) == len(self.attributes)
        assert self.target_attribute in self.attributes
        assert self.target_attribute not in self.input_attributes
        assert set(self.input_attributes).issubset(set(self.attributes))
        if self.check_values_flag:
            # only check if values are provided while initializing DataSet
            [self.check_example(example) for example in self.examples]


    def check_example(self, example):
        if self.values:
            for attr in self.attributes:
                if example[attr] not in self.values:
                    raise ValueError("Not recognized value of {} for attribute {} in {}"
                                     .format(example[attr], attr, example))


    def add_example(self, example):
        self.check_example(example)
        self.examples.append(example)


    def update_values(self):
        self.values = list(map(unique, zip(*self.examples)))


    def remove_examples(self, value=""):
        self.examples = [example for example in examples if value not in example]

    def sanitize(self, example):
        """Copy of the examples with non input_attributes replaced by None"""
        _list_ = []
        for i, attr_i in enumerate(example):
            if i in self.input_attributes:
                _list_.append(attr_i)
            else:
                _list_.append(None)
        return _list_

    def train_test_split(self, test_fraction=0.3, Seed = 99):
        import numpy as np
        
        examples = self.examples
        atrrs = self.attributes
        attr_names = self.attribute_names
        target = self.target_attribute
        input_ = self.input_attributes
        name = self.name 

        np.random.seed(Seed)
        _test_index = np.random.choice(list(range(len(examples))), int(test_fraction * len(examples)), replace=False)

        test_examples = [example for i, example in enumerate(examples) if i in _test_index]

        train_examples = [example for example in examples if example not in test_examples]

        Test_data_set = Data_Set(examples=test_examples,
                                 attributes=atrrs,
                                 attribute_names=attr_names,
                                 target_attribute=target,
                                 input_attributes=input_,
                                 name=name + " Test set",)

        Train_data_set = Data_Set(examples=train_examples,
                                 attributes=atrrs,
                                 attribute_names=attr_names,
                                 target_attribute=target,
                                 input_attributes=input_,
                                 name=name + " Train set",)

        return Train_data_set, Test_data_set
                
        
    def __repr__(self):
        return '<DataSet({}): with {} examples, and {} attributes>'.format(
            self.name, len(self.examples), len(self.attributes))
