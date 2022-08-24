#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "crc64.h"

// ave = 3.517540

void judge(const std::string& truth, const std::string& guess, int feedback[5])
{
	unsigned char used[5] = { 0,0,0,0,0 };

	for (int i = 0; i < 5; i++)
	{
		if (guess[i] == truth[i])
		{
			feedback[i] = 2;
			used[i] = 1;
		}
		else
		{
			feedback[i] = 0;
		}
	}

	for (int i = 0; i < 5; i++)
	{
		if (feedback[i] == 0)
		{
			for (int j = 0; j < 5; j++)
			{
				if (used[j] == 0 && guess[i] == truth[j])
				{
					feedback[i] = 1;
					used[j] = 1;
					break;
				}
			}
		}
	}
}

int encode(const int feedback[5])
{
	return (((feedback[0] * 10 + feedback[1]) * 10 + feedback[2]) * 10 + feedback[3]) * 10 + feedback[4];
}

struct Child
{
	int idx_guess;
	int feedback_code;
	uint64_t key;
};

class SubList : public std::vector<short>
{
public:
	std::vector<Child> children;

	uint64_t hash() const
	{		
		return crc64(1, (const unsigned char*)data(), sizeof(short)*size());
	}
};

typedef std::unordered_map<uint64_t, SubList> SublistDict;

class Traverser
{
public:
	std::vector<std::string> words;

	SublistDict Collection;

	void Split(SubList& word_idxs, int idx_guess, std::queue<uint64_t>& new_keys)
	{
		std::string guess = words[idx_guess];

		SublistDict sublists;
		for (size_t j = 0; j < word_idxs.size(); j++)
		{
			std::string truth = words[word_idxs[j]];
			int feedback[5];
			judge(truth, guess, feedback);

			uint64_t code = (uint64_t)encode(feedback);
			SubList& sub = sublists[code];
			sub.push_back(word_idxs[j]);
		}

		auto iter = sublists.begin();
		while (iter != sublists.end())
		{
			const SubList& sub = iter->second;			
			if (sub.size() > 1)
			{
				uint64_t hash = sub.hash();
			
				auto iter2 = Collection.find(hash);
				if (iter2 == Collection.end())
				{
					Collection[hash] = sub;
					new_keys.push(hash);
				}

				if (sub.size() < word_idxs.size())
				{
					word_idxs.children.push_back({ idx_guess, (int)(iter->first), hash });
				}
			}
			else
			{
				if (sub[0] != idx_guess)
				{
					word_idxs.children.push_back({ idx_guess, (int)(iter->first), 0 });
				}
			}
			iter++;
		}
	}

	void SplitKey(uint64_t key, std::queue<uint64_t>& new_keys)
	{
		SubList& word_idxs = Collection[key];

		for (size_t i = 0; i < word_idxs.size(); i++)
		{
			int idx_guess = word_idxs[i];
			std::string guess = words[idx_guess];

			Split(word_idxs, idx_guess, new_keys);
		}

	}

};


int main()
{
	Traverser trav;
	
	{
		FILE* fp = fopen("wordle.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			trav.words.push_back(word);
		}
		fclose(fp);
	}
	
	SubList full_idxs;
	full_idxs.resize(trav.words.size());
	for (size_t i = 0; i < full_idxs.size(); i++)
	{
		full_idxs[i] = i;
	}
	uint64_t first_hash = full_idxs.hash();
	trav.Collection[first_hash] = full_idxs;


	std::string first_guess = "slate";
	int first_idx = int(std::find(trav.words.begin(), trav.words.end(), first_guess) - trav.words.begin());

	std::queue<uint64_t> key_new_lists;
	trav.Split(trav.Collection[first_hash], first_idx, key_new_lists);

	while (key_new_lists.size() > 0)
	{
		uint64_t key = key_new_lists.front();
		key_new_lists.pop();

		trav.SplitKey(key, key_new_lists);	
	}

	struct Info
	{
		int sel_idx;
		float ave_len;	
		size_t count;		
	};

	std::unordered_map<uint64_t, Info> InfoMap;

	size_t last_count = 0;

	while (true)
	{
		size_t count_resolved = 0;

		auto iter = trav.Collection.begin();
		while (iter != trav.Collection.end())
		{
			SubList& lst = iter->second;

			bool can_solve = true;
			for (int i = 0; i < lst.children.size(); i++)
			{
				uint64_t key = lst.children[i].key;
				if (key != 0)
				{
					if (InfoMap.find(key) == InfoMap.end())
					{
						can_solve = false;
						break;
					}
				}
			}

			if (can_solve)
			{
				struct Sum
				{
					double sum_len = 0.0;
					double sum_weight = 1.0;
				};
				std::unordered_map<int, Sum> sums;

				for (int i = 0; i < lst.children.size(); i++)
				{
					Child& child = lst.children[i];
					Sum& sum = sums[child.idx_guess];
					uint64_t key = child.key;					

					if (key == 0)
					{
						sum.sum_len += 1.0;
						sum.sum_weight += 1.0;
					}
					else
					{
						Info& info = InfoMap[key];
						double weight = (double)info.count;
						sum.sum_len += weight * (double)info.ave_len;
						sum.sum_weight += weight;
					}
				}

				Info info;
				info.count = lst.size();
				float min_ave_len = FLT_MAX;
				auto i_sums = sums.begin();
				while (i_sums != sums.end())
				{
					int sel_idx = i_sums->first;
					Sum& sum = i_sums->second;
					float ave_len = (float)(sum.sum_len / sum.sum_weight);
					if (ave_len < min_ave_len)
					{
						min_ave_len = ave_len;
						info.sel_idx = sel_idx;										
					}
					i_sums++;
				}
				info.ave_len = min_ave_len + 1.0f;
				InfoMap[iter->first] = info;

				count_resolved++;
			}

			iter++;
		}

		printf("%llu/%llu\n", count_resolved, trav.Collection.size());
		if (count_resolved == trav.Collection.size()) break;

		if (count_resolved == last_count)
		{
			break;
		}

		last_count = count_resolved;

	}
    
	std::queue<uint64_t> trav_queue;
	trav_queue.push(first_hash);
	while (trav_queue.size() > 0)
	{
		uint64_t key = trav_queue.front();
		trav_queue.pop();

		Info& info = InfoMap[key];
		SubList& lst = trav.Collection[key];

		int idx_next = info.sel_idx;

		char filename[100];
		sprintf(filename, "data/%llx", key);
		FILE* fp = fopen(filename, "w");
		fprintf(fp, "%s\n", trav.words[idx_next].c_str());
		
		for (size_t i = 0; i < lst.children.size(); i++)
		{
			Child& child = lst.children[i];
			if (child.idx_guess == idx_next)
			{
				fprintf(fp, "%05d %llx\n", child.feedback_code, child.key);
				if (child.key != 0)
				{
					trav_queue.push(child.key);
				}
			}
		}
		fclose(fp);		
	}

	printf("data/%llx\n", first_hash);
	
	return 0;
}