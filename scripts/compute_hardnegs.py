import datasets
import numpy as np

from colpali_engine.models.paligemma_colbert_architecture import BiPaliMean
from transformers import AutoProcessor
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from colpali_engine.utils.colpali_processing_utils import process_images
from colpali_engine.utils.dataset_transformation import load_train_set

train_set = load_train_set()


COMPUTE_EMBEDDINGS = False
COMPUTE_HARDNEGS = False

if COMPUTE_HARDNEGS or COMPUTE_EMBEDDINGS:
    model_name = "./models/bipali"

    model = BiPaliMean.from_pretrained("./models/paligemma-3b-mix-448",
                                       torch_dtype=torch.bfloat16,
                                       device_map="cuda").eval()
    print("Add adapter")
    model.load_adapter(model_name)
    print("Model loaded")
    model = model.to("cuda")
    print("Processor")
    processor = AutoProcessor.from_pretrained(model_name)

if COMPUTE_EMBEDDINGS:
    # select images -> load_from_pdf(<pdf_path>),  load_from_image_urls(["<url_1>"]), load_from_dataset(<path>)

    print("Loading images")
    print("Images loaded")

    document_set = train_set["train"]
    print("Filtering dataset")
    print(document_set)
    initial_list = document_set['image_filename']
    _, unique_indices = np.unique(initial_list, return_index=True, axis=0)
    filtered_dataset = document_set.select(unique_indices.tolist())
    filtered_dataset = filtered_dataset.map(lambda example: {'image': example['image'], 'image_filename': example['image_filename']}, num_proc=16)
    # keep only column image and image_filename and source if it exists
    cols_to_remove = [col for col in filtered_dataset.column_names if col not in ["image", "image_filename"]]
    filtered_dataset = filtered_dataset.remove_columns(cols_to_remove)
    # save it
    print("Saving filtered dataset")
    print(filtered_dataset)
    filtered_dataset.save_to_disk("data_dir/filtered_dataset", max_shard_size="200MB")

    print("Processing images")
    # run inference - docs
    dataloader = DataLoader(
        filtered_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, [a["image"] for a in x]),
    )
    print("Computing embeddings")

    ds = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    ds = torch.stack(ds)

    # save embeddings
    torch.save(ds, "data_dir/filtered_dataset_embeddings.pt")

if not COMPUTE_EMBEDDINGS:
    ds = torch.load("data_dir/filtered_dataset_embeddings.pt")


if COMPUTE_HARDNEGS:
    # compute hard negatives
    ds = ds.to("cuda")

    from colpali_engine.utils.colpali_processing_utils import process_queries

    # iterate on the train set

    mined_hardnegs = []

    for i in tqdm(range(0, len(train_set["train"]), 8)):
        samples = train_set["train"][i : i + 8]
        batch_query = process_queries(processor, samples["query"], train_set["train"][i]["image"])
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)

        # compute scores
        scores = torch.einsum("bd,cd->bc", embeddings_query, ds)
        # get top 100 indexes
        top100 = scores.topk(100, dim=1).indices
        # indices to list
        top100 = top100.tolist()
        # append to mined_hardnegs
        mined_hardnegs.extend(top100)

    # save mined hardnegs as txt
    with open("data_dir/mined_hardnegs_filtered.txt", "w") as f:
        for item in mined_hardnegs:
            f.write("%s\n" % item)


with open("data_dir/mined_hardnegs_filtered.txt") as f:
    mined_hardnegs = f.readlines()


filtered_dataset = datasets.load_from_disk("data_dir/filtered_dataset")
filenames = list(filtered_dataset['image_filename'])


def mapper_fn(example, idx):
    tmp = {"negs": [int(x) for x in mined_hardnegs[idx][1:-2].strip().split(",")],
           "query": example["query"],
           "gold_index": filenames.index(example["image_filename"])}

    tmp["gold_in_top_100"] = tmp["gold_index"] in tmp["negs"]
    # remove gold index from negs if it is there
    if tmp["gold_in_top_100"]:
        tmp["negs"].remove(tmp["gold_index"])
    return tmp


final_dataset = train_set["train"].map(mapper_fn, with_indices=True, num_proc=16)
# drop image
final_dataset = final_dataset.remove_columns("image")
final_dataset.save_to_disk("data_dir/final_dataset")
