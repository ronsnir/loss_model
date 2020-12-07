import setuptools
 
with open("README.md", "r") as fh:
   long_description = fh.read()
 
setuptools.setup(
   name="loss_model",
   version="0.0.1",
   maintainer='US Underwriting Team',
   description="A model that predicts the losses",
   long_description=long_description,
   long_description_content_type="text/markdown",
   packages=setuptools.find_packages(),
   python_requires='>=3.6',
)
