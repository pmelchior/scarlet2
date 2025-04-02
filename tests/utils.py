import pip
import importlib

package_name = "scarlet_test_data"
package_link = "git+https://github.com/astro-data-lab/scarlet-test-data.git"

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

def import_scarlet_test_data():
    try:
        import scarlet_test_data

    except ImportError:
        install(package_link)
    
    finally:
        globals()[package_name] = importlib.import_module(package_name)

if __name__=="__main__":
    import_scarlet_test_data()
    from scarlet_test_data import data_path
    print(data_path)
