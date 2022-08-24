#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>

#include <cstdint>
#include <chrono>

inline uint64_t time_micro_sec()
{
	std::chrono::time_point<std::chrono::system_clock> tpSys = std::chrono::system_clock::now();
	std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> tpMicro
		= std::chrono::time_point_cast<std::chrono::microseconds>(tpSys);
	return tpMicro.time_since_epoch().count();
}

inline uint64_t time_milli_sec()
{
	return (time_micro_sec() + 500) / 1000;
}

inline double time_sec()
{
	return (double)time_micro_sec() / 1000000.0;
}

inline bool exists_test(const char* name)
{
	if (FILE* file = fopen(name, "r"))
	{
		fclose(file);
		return true;
	}
	else
	{
		return false;
	}
}



// cuda
void h_set_words(const std::vector<std::string>& all_words);
void h_guess(const std::vector<std::string>& all_words, int num_words, const std::string* words, int* rounds, const std::string& start_word);
void h_guess_all(const std::vector<std::string>& all_words, int* rounds, const std::string& start_word);
void h_guess_matrix(int num_words, int* rounds);

// about: 4.041576
// trace: 3.802945


int main()
{
	// sort words
	/* {
		std::unordered_set<std::string> word_set;
		{
			FILE* fp = fopen("wordle.txt", "r");
			char line[100];
			while (fgets(line, 100, fp))
			{
				char word[100];
				sscanf(line, "%s", word);
				word_set.insert(word);
			}
			fclose(fp);
		}		

		struct WordFreq
		{
			std::string word;
			int freq;
		};

		std::vector<WordFreq> words;

		{
			FILE* fp = fopen("unigram_freq.csv", "r");
			char line[100];
			while (fgets(line, 100, fp))
			{
				std::string s_line = line;
				int pos = s_line.find(',');
				std::string word = s_line.substr(0, pos);

				if (word.length() == 5)
				{					
					if (word_set.find(word) != word_set.end())
					{
						int count = atoi(line + pos + 1);
						words.push_back({ word, count });						
					}
				}
			}
			fclose(fp);
		}
		

		std::sort(words.begin(), words.end(), [](const WordFreq& a, const WordFreq& b)
		{
			return a.freq > b.freq;
		});

		{
			FILE* fp = fopen("wordle_freq_sorted.txt", "w");
			for (size_t i = 0; i < words.size(); i++)
			{
				fprintf(fp, "%s\n", words[i].word.c_str());
			}

			fclose(fp);
		}

	}*/

	std::vector<std::string> words;
	{
		FILE* fp = fopen("wordle_freq_sorted.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			words.push_back(word);
		}
		fclose(fp);
	}

	/* {
		h_set_words(words);

		std::string truth = "merit";
		int rounds;
		h_guess(words, 1, &truth, &rounds, "about");

		printf("%d\n", rounds);
	}*/

	/* {
		h_set_words(words);

		FILE* fp = fopen("begins.csv", "w");
		for (size_t i = 0; i < words.size(); i++)
		{
			std::string begin = words[i];

			std::vector<int> rounds(words.size());
			h_guess_all(words, rounds.data(), begin);

			size_t total_guess = 0;
			for (size_t j = 0; j < words.size(); j++)
			{
				total_guess += (size_t)rounds[j];
			}
			double ave_guess = (double)(total_guess) / (double)words.size();
			fprintf(fp, "%s, %f\n", begin.c_str(), ave_guess);

			printf("%s, %f   %d/%d\n", begin.c_str(), ave_guess, i, words.size());

		}
		fclose(fp);
	}*/


	for (int iter = 0; iter<5; iter++)
	{
		h_set_words(words);

		std::vector<int> rounds(words.size() * words.size());
		h_guess_matrix((int)words.size(), rounds.data());

		struct Index
		{
			int idx;
			double ave;
		};

		std::vector<Index> index(words.size());
		
		for (size_t i = 0; i < words.size(); i++)
		{		
			size_t total_guess = 0;
			for (size_t j = 0; j < words.size(); j++)
			{
				total_guess += (size_t)rounds[j + i * words.size()];
			}
			index[i].idx = i;
			index[i].ave =  (double)(total_guess) / (double)words.size();
		}

		
		std::sort(index.begin(), index.end(), [](const Index& a, const Index& b)
		{
			return a.ave < b.ave;
		});

		printf("%s %f\n", words[index[0].idx].c_str(), index[0].ave);

		std::vector<std::string> lst_words(words.size());
		for (size_t i = 0; i < words.size(); i++)
		{
			int j = index[i].idx;
			lst_words[i] = words[j];
		}

		words = lst_words;

		if (iter == 3)
		{
			FILE* fp = fopen("wordle_machine_optimized.txt", "w");
			for (size_t i = 0; i < words.size(); i++)
			{
				std::string word = words[i];
				fprintf(fp, "%s\n", word.c_str());

			}
			fclose(fp);
		}
		else if (iter == 4)
		{
			FILE* fp = fopen("op_begins.csv", "w");
			for (size_t i = 0; i < words.size(); i++)
			{
				std::string begin = words[i];
				double ave_guess = index[i].ave;
				fprintf(fp, "%s, %f\n", begin.c_str(), ave_guess);

			}
			fclose(fp);
		}
	}


	return 0;
}

