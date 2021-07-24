import json


if __name__ == "__main__":

  with open("retrieval_database/metadata.json") as f:
    metadata = json.load(f)
  with open("retrieval_database/annotations.json") as f:
    annotations = json.load(f)

  matches = []
  #sum_diffs = []
  #abs_sum_diffs = []
  #abs_diffs_allcoinsfound = []
  relative_errors = []
  relative_errors_allcoinsfound = []
  max_missing = 5
  for img_name in metadata:
    coins_present = int(metadata[img_name]["num_coins"])
    coins_recognized = int(annotations[img_name]["num_coins"]) 
    matches.append(abs(coins_present - coins_recognized) <= max_missing)
    
    sum_present = int(metadata[img_name]["sum"])
    sum_recognized = int(annotations[img_name]["sum"]) 
    #sum_diffs.append(sum_present - sum_recognized)
    #abs_sum_diffs.append(abs(sum_present - sum_recognized))

    relative_error = abs(sum_recognized - sum_present)/sum_present
    if coins_present == coins_recognized:
      #abs_diffs_allcoinsfound.append(abs(sum_present - sum_recognized))
      relative_errors_allcoinsfound.append(relative_error)

    relative_errors.append(relative_error)


  print(f"There are {sum(matches)} images where at most {max_missing} coin(s) are missing")
  #print(f"{min(sum_diffs)}\t{max(sum_diffs)}\t{sum(sum_diffs) / len(sum_diffs)}")
  #print(f"{min(abs_sum_diffs)}\t{max(abs_sum_diffs)}\t{sum(abs_sum_diffs) / len(abs_sum_diffs)}")
  #print(f"{min(abs_diffs_allcoinsfound)}\t{max(abs_diffs_allcoinsfound)}\t{sum(abs_diffs_allcoinsfound) / len(abs_diffs_allcoinsfound)}")
  print(f"average relative errors over all images: {sum(relative_errors)/len(relative_errors)}")
  print(f"average relative error over all images where all coins are found: {sum(relative_errors_allcoinsfound) / len(relative_errors_allcoinsfound)}")
