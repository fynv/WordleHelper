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
void h_guess_matrix(int num_words, int* rounds);

int main()
{	
	std::vector<std::string> words;
	{
		FILE* fp = fopen("wordle_machine_optimized.txt", "r");		
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			words.push_back(word);		
		}
		fclose(fp);
	}

	bool first = true;

	std::unordered_set<char> exclude_sets[5];	
	while(true)
	{
		std::string guess;

		if (first)
		{
			guess = "slate";
			first = false;
		}
		else
		{
			h_set_words(words);
			std::vector<int> rounds(words.size() * words.size());
			h_guess_matrix((int)words.size(), rounds.data());

			double best_len = FLT_MAX;
			size_t best_i = -1;

			for (size_t i = 0; i < words.size(); i++)
			{
				size_t total_length = 0;
				for (size_t j = 0; j < words.size(); j++)
				{
					total_length += (size_t)rounds[j + i * words.size()];
				}
				double ave_length = (double)(total_length) / (double)words.size();
				if (ave_length < best_len)
				{
					best_len = ave_length;
					best_i = i;
				}
			}
			guess = words[best_i];
		}

		printf("%s\n", guess.c_str());

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

		unsigned char min_counts[26]; memset(min_counts, 0, 26);
		unsigned char max_counts[26]; memset(max_counts, 5, 26);
		for (int i = 0; i < 5; i++)
		{
			char c = guess[i];
			int j = feedback[i];

			if (j == 1)
			{
				min_counts[c - 'a']++;
				exclude_sets[i].insert(c);
			}
			else if (j == 2)
			{
				min_counts[c - 'a']++;
				if (exclude_sets[i].size() < 25)
				{
					for (int k = 0; k < 26; k++)
					{
						char c2 = 'a' + k;
						if (c2 != c)
						{
							exclude_sets[i].insert(c2);
						}
					}
				}
			}
			else
			{
				exclude_sets[i].insert(c);
			}
		}

		for (int i = 0; i < 5; i++)
		{
			char c = guess[i];
			int j = feedback[i];
			if (j != 1 && j != 2)
			{
				max_counts[c - 'a'] = min_counts[c - 'a'];
			}
		}

		for (size_t i = 0; i < words.size(); i++)
		{
			std::string word = words[i];
			bool remove = false;
			unsigned char counts[26]; memset(counts, 0, 26);

			for (int j = 0; j < 5; j++)
			{
				char c = word[j];
				auto iter = exclude_sets[j].find(c);
				if (iter != exclude_sets[j].end())
				{
					remove = true;					
					break;
				}
				counts[c - 'a']++;
			}

			if (!remove)
			{
				for (int k = 0; k < 26; k++)
				{
					unsigned char min_count = min_counts[k];
					unsigned char max_count = max_counts[k];
					unsigned char count = counts[k];
					if (count<min_count || count>max_count)
					{
						remove = true;
						break;
					}
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

/*void judge(const std::string& truth, const std::string& guess, int feedback[5])
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

int main()
{
	std::vector<std::string> all_words;
	{
		FILE* fp = fopen("wordle_freq.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			std::string s_line = line;
			int pos = s_line.find(',');
			std::string word = s_line.substr(0, pos);
			all_words.push_back(word);
		}
		fclose(fp);
	}

	size_t total_guess_count = 0;
	for (size_t true_i = 0; true_i < all_words.size(); true_i++)
	{
		std::string truth = all_words[true_i];
		
		std::vector<std::string> words = all_words;
		
		int guess_count = 0;
		std::unordered_set<char> exclude_sets[5];
		while (true)
		{
			std::string guess;
			if (guess_count == 0)
			{
				guess = "raise";
			}
			else
			{
				h_set_words(words);
				std::vector<int> rounds(words.size() * words.size());
				h_guess_matrix((int)words.size(), rounds.data());

				double best_len = FLT_MAX;
				size_t best_i = -1;

				for (size_t i = 0; i < words.size(); i++)
				{
					size_t total_length = 0;
					for (size_t j = 0; j < words.size(); j++)
					{
						total_length += (size_t)rounds[j + i * words.size()];
					}
					double ave_length = (double)(total_length) / (double)words.size();
					if (ave_length < best_len)
					{
						best_len = ave_length;
						best_i = i;
					}
				}
				guess = words[best_i];
			}
			guess_count++;

			int feedback[5];
			judge(truth, guess, feedback);

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

			unsigned char min_counts[26]; memset(min_counts, 0, 26);
			unsigned char max_counts[26]; memset(max_counts, 5, 26);
			for (int i = 0; i < 5; i++)
			{
				char c = guess[i];
				int j = feedback[i];

				if (j == 1)
				{
					min_counts[c - 'a']++;
					exclude_sets[i].insert(c);
				}
				else if (j == 2)
				{
					min_counts[c - 'a']++;
					if (exclude_sets[i].size() < 25)
					{
						for (int k = 0; k < 26; k++)
						{
							char c2 = 'a' + k;
							if (c2 != c)
							{
								exclude_sets[i].insert(c2);
							}
						}
					}
				}
				else
				{
					exclude_sets[i].insert(c);
				}
			}

			for (int i = 0; i < 5; i++)
			{
				char c = guess[i];
				int j = feedback[i];
				if (j != 1 && j != 2)
				{
					max_counts[c - 'a'] = min_counts[c - 'a'];
				}
			}

			for (size_t i = 0; i < words.size(); i++)
			{
				std::string word = words[i];
				bool remove = false;
				unsigned char counts[26]; memset(counts, 0, 26);

				for (int j = 0; j < 5; j++)
				{
					char c = word[j];
					auto iter = exclude_sets[j].find(c);
					if (iter != exclude_sets[j].end())
					{
						remove = true;
						break;
					}
					counts[c - 'a']++;
				}

				if (!remove)
				{
					for (int k = 0; k < 26; k++)
					{
						unsigned char min_count = min_counts[k];
						unsigned char max_count = max_counts[k];
						unsigned char count = counts[k];
						if (count<min_count || count>max_count)
						{
							remove = true;
							break;
						}
					}
				}
				if (remove)
				{
					words.erase(words.begin() + i);
					i--;
				}
			}

		}

		printf("%s %d  %d/%d\n", truth.c_str(), guess_count, true_i, (int)all_words.size());

		total_guess_count += (size_t)guess_count;
	}

	double ave_count = (double)(total_guess_count) / (double)all_words.size();
	printf("ave: %f\n", ave_count);

	return 0;
}*/
