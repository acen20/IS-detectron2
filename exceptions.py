
# Define error and their messages
def throw_error(key, arg = None):

  errors = {
      'MemoryError':{
          'message': f'The host ran out of memory. Try to reduce the batch size'
      },
      'FileNotFoundError':{
          'message': f'Unable to locate the file {arg}'
      },
      'RuntimeError':{
      		'message': f'{arg}'
      },
      'FloatingPointError':{
      		'message': f'Training has diverged. Try using a smaller learning rate'
      },
      'error':{
      		'message': f'Please input valid data'
      },
      'AttributeError':{
      		'message': f'Please input a valid image format'
      },
      'Exception':{
      		'message':f'{arg}'
      },
      'Default':{
      	'message':f'{arg}'
      }
  }
  
  if key not in errors.keys():
    key='Default'
  return errors[key]


## Handle Exceptions
def handle_exception(e):
	error_type = type(e).__name__
	
	## Errors having arguments should be defined with if block like this
	if error_type == "FileNotFoundError":
		return throw_error(error_type, e.filename)
	if error_type == "Exception":
		return throw_error(error_type, str(e))
	if error_type == "RuntimeError":
		return throw_error(error_type, str(e))
	
	## If there are no arguments, use this
	else:
		return throw_error(error_type, e)
	
