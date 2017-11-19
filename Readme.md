Novel To Vectors
===

# What is this script?
This script transform arbitrary novel in txt format into sentence vectors. For example
```python
text = "I like this script. But, this does not look like a tool."
```
will be transformed into
```
data = 
array([[  4.,   2.,   8.,   3.,   7.,   0.,   0.,   0.,   0.]
,[  6.,   8.,   9.,  10.,  11.,   2.,   1.,   5.,   7.]])
	   
dictionary = 
{0: 'PAD',
 1: u'a',
 2: u'like',
 3: u'script',
 4: u'i',
 5: u'tool',
 6: u'but',
 7: u'.',
 8: u'this',
 9: u'does',
 10: u'not',
 11: u'look'}
 
length = [5, 9]

```

# How to use?
```
python novel_extractor.py -i warpeace_input.txt -o data
```

# Why?
This is a standard format to feed in to RNN neural network

# Format?
Pickle
