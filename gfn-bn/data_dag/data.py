import urllib.request
import gzip

from numpy.random import default_rng
from data_dag.graph import sample_erdos_renyi_linear_gaussian,sample_from_linear_gaussian
def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()
    with open(filename, 'wb') as f:
        f.write(file_content)
    
    return filename


def get_data(name, args,rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(graph,num_samples=args.num_samples,rng=rng)
        score = 'bge'
    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score



