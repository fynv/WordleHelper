#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include "crc64.h"

// 3.647 (wrong)
// unbiased: 3.636285
// biased: 3.468683

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

int main()
{
	double log2guess = 2.638445 / log(2315.0);

	std::vector<std::string> words;
	std::vector<std::string> alloweds;
	{
		FILE* fp = fopen("wordle.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			words.push_back(word);
		}
		fclose(fp);
	}

	{
		alloweds = words;
		FILE* fp = fopen("wordle-allowed-guesses.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			alloweds.push_back(word);
		}
		fclose(fp);
	}

	while (true)
	{
		std::string best;
		if (words.size() > 1)
		{
			std::vector<double> guesses(alloweds.size());

			#pragma omp parallel for			
			for (int i = 0; i < (int)alloweds.size(); i++)
			{
				std::string guess = alloweds[i];
				std::unordered_map<int, int> counts;

				bool has_truth = false;
				for (size_t j = 0; j < words.size(); j++)
				{
					std::string truth = words[j];
					int feedback[5];
					judge(truth, guess, feedback);
					int code = encode(feedback);
					int& count = counts[code];
					count++;
					if (code == 22222)
					{
						has_truth = true;
					}
				}

				double time_guess = 0.0;
				auto iter = counts.begin();
				while (iter != counts.end())
				{
					double count = (double)iter->second;
					time_guess += (log(count) * log2guess + 1.0) * count;
					iter++;
				}
				if (has_truth)
				{
					time_guess -= 1.0;
				}
				time_guess /= (double)words.size();

				guesses[i] = time_guess;
			}

			double min_guesses = FLT_MAX;
			for (size_t i = 0; i < alloweds.size(); i++)
			{
				double time_guess = guesses[i];
				if (time_guess < min_guesses)
				{
					min_guesses = time_guess;
					best = alloweds[i];
				}
			}
		}
		else
		{
			best = words[0];
		}

		printf("%s\n", best.c_str());

		while (true)
		{
			char line[100];
			scanf("%s", line);
			if (strlen(line) >= 5)
			{
				std::string s_line = line;
				best = s_line.substr(0, 5);
				break;
			}
		}

		int feedback[5];
		while (true)
		{
			char line[100];
			scanf("%s", line);
			if (strlen(line) >= 5)
			{
				for (int i = 0; i < 5; i++)
				{
					if (line[i] == '1') feedback[i] = 1;
					else if (line[i] == '2') feedback[i] = 2;
					else feedback[i] = 0;
				}
				break;
			}
		}

		bool match = true;
		for (int i = 0; i < 5; i++)
		{
			if (feedback[i] != 2)
			{
				match = false;
				break;
			}
		}
		if (match) break;

		for (size_t i = 0; i < words.size(); i++)
		{
			std::string word = words[i];

			bool remove = false;

			int feedback2[5];
			judge(word, best, feedback2);
			for (int j = 0; j < 5; j++)
			{
				if (feedback2[j] != feedback[j])
				{
					remove = true;
					break;
				}
			}

			if (remove)
			{
				words.erase(words.begin() + i);
				i--;
			}
		}
	}
	
	return 0;
}

/*int main()
{
	double log2guess = 2.638445 / log(2315.0);

	std::vector<std::string> words;
	std::vector<std::string> alloweds;
	{
		FILE* fp = fopen("wordle.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			words.push_back(word);
		}
		fclose(fp);
	}

	{
		alloweds = words;
		FILE* fp = fopen("wordle-allowed-guesses.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			alloweds.push_back(word);
		}
		fclose(fp);
	}

	std::vector<std::string> guesses;

	while (guesses.size() < 5)
	{
		std::vector<double> values(alloweds.size(), FLT_MAX);

		#pragma omp parallel for			
		for (int i = 0; i < (int)alloweds.size(); i++)
		{
			std::string guess = alloweds[i];
			bool guessed = false;
			for (int j = 0; j < (int)guesses.size(); j++)
			{
				if (guess == guesses[j])
				{
					guessed = true;
					break;
				}
			}
			if (guessed) continue;

			std::unordered_map<uint64_t, int> counts;
			int truth_count = 0;
			for (int k = 0; k < (int)words.size(); k++)
			{
				std::string truth = words[k];
				
				std::vector<int> codes(guesses.size() + 1);
				for (int j = 0; j < (int)guesses.size(); j++)
				{
					int feedback[5];
					judge(truth, guesses[j], feedback);
					int code = encode(feedback);
					if (code == 22222)
					{
						truth_count++;
					}
					codes[j] = code;
					
				}
				{
					int feedback[5];
					judge(truth, guess, feedback);
					int code = encode(feedback);
					if (code == 22222)
					{
						truth_count++;
					}
					codes[guesses.size()] = code;
				}
				uint64_t hash = crc64(1, (const unsigned char*)codes.data(), sizeof(int) * codes.size());
				int& count = counts[hash];
				count++;
			}
			double max_count = 0;
			double time_guess = 0.0;
			auto iter = counts.begin();
			while (iter != counts.end())
			{
				double count = (double)iter->second;
				time_guess += (log(count) * log2guess + 1.0) * count;
				if (count > max_count) max_count = count;
				iter++;
			}
			time_guess -= (double)truth_count;
			time_guess /= (double)words.size();
			// values[i] = time_guess;
			values[i] = max_count;
		}

		int selected_id = -1;
		double min_value = FLT_MAX;
		for (int i = 0; i < (int)alloweds.size(); i++)
		{
			double value = values[i];
			if (value < min_value)
			{
				min_value = value;
				selected_id = i;
			}
		}

		std::string selected = alloweds[selected_id];
		guesses.push_back(selected);
		printf("%s\n", selected.c_str());
	}
}*/


/*
struct Child
{	
	int feedback_code;
	uint64_t key;
};

class SubList : public std::vector<short>
{
public:
	std::string guess;
	std::vector<Child> children;

	uint64_t hash() const
	{
		return crc64(1, (const unsigned char*)data(), sizeof(short) * size());
	}
};

typedef std::unordered_map<uint64_t, SubList> SublistDict;

class Guesser
{
public:
	std::vector<std::string> words;
	std::vector<std::string> alloweds;

	SublistDict Collection;

	void Split(SubList& word_idxs, std::string guess, std::queue<uint64_t>& new_keys)
	{
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
					word_idxs.children.push_back({ (int)(iter->first), hash });
				}
			}
			iter++;
		}
	}

	void Guess(SubList& word_idxs, std::queue<uint64_t>& new_keys)
	{
		double log2guess = 2.638445 / log(2315.0);

		std::string best;
		
		std::vector<double> guesses(alloweds.size());
		
		#pragma omp parallel for			
		for (int i = 0; i < (int)alloweds.size(); i++)
		{
			std::string guess = alloweds[i];
			std::unordered_map<int, int> counts;

			bool has_truth = false;
			for (size_t j = 0; j < word_idxs.size(); j++)
			{
				std::string truth = words[word_idxs[j]];
				int feedback[5];
				judge(truth, guess, feedback);
				int code = encode(feedback);
				int& count = counts[code];
				count++;
				if (code == 22222)
				{
					has_truth = true;
				}
			}

			double time_guess = 0.0;
			auto iter = counts.begin();
			while (iter != counts.end())
			{
				double count = (double)iter->second;
				time_guess += (log(count) * log2guess + 1.0) * count;
				iter++;
			}
			if (has_truth)
			{
				time_guess -= 1.0;
			}
			time_guess /= (double)words.size();

			guesses[i] = time_guess;
		}

		double min_guesses = FLT_MAX;
		for (size_t i = 0; i < alloweds.size(); i++)
		{
			double time_guess = guesses[i];
			if (time_guess < min_guesses)
			{
				min_guesses = time_guess;
				best = alloweds[i];
			}
		}

		word_idxs.guess = best;

		Split(word_idxs, best, new_keys);
	}

	void Guess(uint64_t key, std::queue<uint64_t>& new_keys)
	{
		SubList& word_idxs = Collection[key];
		Guess(word_idxs, new_keys);
	}
};

int main()
{
	Guesser guesser;

	{
		FILE* fp = fopen("wordle.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			guesser.words.push_back(word);
		}
		fclose(fp);
	}

	{
		guesser.alloweds = guesser.words;
		FILE* fp = fopen("wordle-allowed-guesses.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			guesser.alloweds.push_back(word);
		}
		fclose(fp);
	}


	SubList full_idxs;
	full_idxs.resize(guesser.words.size());
	for (size_t i = 0; i < full_idxs.size(); i++)
	{
		full_idxs[i] = i;
	}

	uint64_t first_hash = full_idxs.hash();
	guesser.Collection[first_hash] = full_idxs;

	std::queue<uint64_t> key_new_lists;
	guesser.Guess(first_hash, key_new_lists);


	while (key_new_lists.size() > 0)
	{
		uint64_t key = key_new_lists.front();
		key_new_lists.pop();

		guesser.Guess(key, key_new_lists);
	}

	auto iter = guesser.Collection.begin();
	while (iter != guesser.Collection.end())
	{
		uint64_t key = iter->first;
		SubList& lst = iter->second;

		char filename[100];
		sprintf(filename, "data1/%llx", key);
		FILE* fp = fopen(filename, "w");

		std::string guess = lst.guess;
		fprintf(fp, "%s\n", guess.c_str());

		for (size_t i = 0; i < lst.children.size(); i++)
		{
			Child& child = lst.children[i];			
			fprintf(fp, "%05d %llx\n", child.feedback_code, child.key);
		}

		fclose(fp);

		iter++;
	}

	printf("data/%llx\n", first_hash);

	return 0;
}*/

/*void read_opt(const char* fn_opt, std::string& opt_suggestion, std::unordered_map<int, std::string>& opt_dict)
{
	std::string path_opt = std::string("data1/") + fn_opt;
	FILE* fp = fopen(path_opt.c_str(), "r");

	char line[1024];
	fgets(line, 1024, fp);
	char word[100];
	sscanf(line, "%s", word);
	opt_suggestion = word;

	opt_dict.clear();
	while (fgets(line, 1024, fp))
	{		
		std::string s_line = line;
		size_t pos = s_line.find(' ');
		if (pos == std::string::npos) continue;
		int code;
		sscanf(line, "%d", &code);			
		sscanf(line + pos + 1, "%s", word);		
		opt_dict[code] = word;
	}

	fclose(fp);

}

int main()
{
	double log2guess = 2.638445 / log(2315.0);

	std::vector<std::string> all_words;
	std::vector<std::string> alloweds;
	{
		FILE* fp = fopen("wordle.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			all_words.push_back(word);
		}
		fclose(fp);
	}

	{
		alloweds = all_words;
		FILE* fp = fopen("wordle-allowed-guesses.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			alloweds.push_back(word);
		}
		fclose(fp);
	}

	int max_guess_count = 0;
	double ave_guesses = 0.0;

	for (size_t i_truth = 0; i_truth < all_words.size(); i_truth++)
	{
		std::string truth_i = all_words[i_truth];
		std::vector<std::string> words = all_words;

		std::string fn_opt = "d64d8a103311a203";

		int guess_count = 0;
		while (true)
		{		

			std::string best;
			std::unordered_map<int, std::string> opt_map;

			if (words.size() > 1)
			{			
				read_opt(fn_opt.c_str(), best, opt_map);
			}
			else
			{
				best = words[0];
			}

			int feedback[5];
			judge(truth_i, best, feedback);
			int code = encode(feedback);

			guess_count++;

			if (code==22222) break;

			for (size_t i = 0; i < words.size(); i++)
			{
				std::string word = words[i];

				bool remove = false;

				int feedback2[5];
				judge(word, best, feedback2);
				int code2 = encode(feedback2);

				if (code2!=code)
				{
					words.erase(words.begin() + i);
					i--;
				}
			}

			if (words.size() > 1)
			{
				fn_opt = opt_map[code];
			}
		}

		ave_guesses += (double)guess_count;
		if (guess_count > max_guess_count)
		{
			max_guess_count = guess_count;
			printf("Max: ");
		}

		printf("%s %d\n", truth_i.c_str(), guess_count);
	}
	ave_guesses /= (double)all_words.size();

	printf("%f\n", ave_guesses);
	printf("%d\n", max_guess_count);

	return 0;
}*/

