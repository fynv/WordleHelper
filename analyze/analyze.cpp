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

class Game
{
public:
	static std::vector<std::string> all_words;

	char truth[5];
	std::string begin = "about";

	void judge(const char guess[5], int feedback[5])
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

	int guess()
	{
		std::vector<std::string> options = all_words;
		std::unordered_set<char> exclude_sets[5];

		int rounds = 0;
		while (options.size() > 0)
		{
			char guess[5];
			if (rounds == 0)
			{
				memcpy(guess, begin.c_str(), 5);
			}
			else
			{
				memcpy(guess, options[0].c_str(), 5);
			}

			rounds++;

			int feedback[5];
			judge(guess, feedback);			

			/*for (int i = 0; i < 5; i++)
			{
				printf("%c", guess[i]);
			}
			printf("\n");
			for (int i = 0; i < 5; i++)
			{
				printf("%d", feedback[i]);
			}
			printf("\n");*/

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
					min_counts[c- 'a']++;
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

			for (size_t i = 0; i < options.size(); i++)
			{
				std::string word = options[i];
				bool remove = false;
				unsigned char counts[26]; memset(counts, 0, 26);

				for (int j = 0; j < 5; j++)
				{
					char c = word[j];
					auto iter = exclude_sets[j].find(c);
					if (iter!= exclude_sets[j].end())
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
					options.erase(options.begin() + i);
					i--;
				}
			}
		}

		return rounds;
	}

};

std::vector<std::string> Game::all_words;

int main()
{
	std::vector<int> freqs;
	{
		// FILE* fp = fopen("wordle_freq.txt", "r");		
		FILE* fp = fopen("op_begins.csv", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			std::string s_line = line;
			int pos = s_line.find(',');
			std::string word = s_line.substr(0, pos);			
			Game::all_words.push_back(word);

			/*int count = atoi(line + pos + 1);
			freqs.push_back(count);*/
		}		
		fclose(fp);
	}

	/* {
		FILE* fp = fopen("tries.csv", "w");

		for (size_t i = 0; i < Game::all_words.size(); i++)
		{
			std::string truth = Game::all_words[i];
			Game game;
			memcpy(game.truth, truth.c_str(), 5);
			int round = game.guess();

			float interest = (float)round / (10.0f - logf((float)Game::freqs[i]) / logf(10.0f));
			fprintf(fp, "%s, %d, %d, %f\n", truth.c_str(), Game::freqs[i], round, interest);
		}

		fclose(fp);
	}*/

	/*FILE* fp = fopen("begins.csv", "w");
	for (size_t i = 0; i < Game::all_words.size(); i++)
	{		
		std::string begin = Game::all_words[i];

		size_t total_guess = 0;
		for (size_t j = 0; j < Game::all_words.size(); j++)
		{
			std::string truth = Game::all_words[j];

			Game game;
			game.begin = begin;
			memcpy(game.truth, truth.c_str(), 5);
			int round = game.guess();
			total_guess += (size_t)round;
		}

		double ave_guess = (double)(total_guess) / (double)Game::all_words.size();
		fprintf(fp, "%s, %f\n", begin.c_str(), ave_guess);
		
		printf("%s, %f   %d/%d\n", begin.c_str(), ave_guess, i, Game::all_words.size());

	}
	fclose(fp);*/

	/* {

		size_t total_guess = 0;
		for (size_t j = 0; j < Game::all_words.size(); j++)
		{
			std::string truth = Game::all_words[j];

			Game game;			
			memcpy(game.truth, truth.c_str(), 5);
			int round = game.guess();
			total_guess += (size_t)round;
		}

		double ave_guess = (double)(total_guess) / (double)Game::all_words.size();
		printf("%f\n", ave_guess);

	}*/

	/*h_set_words(Game::all_words);

	std::string truth = "merit";
	int rounds;
	h_guess(Game::all_words, 1, &truth, &rounds, "about");

	printf("%d\n", rounds);*/

	/*h_set_words(Game::all_words);

	FILE* fp = fopen("begins.csv", "w");
	for (size_t i = 0; i < Game::all_words.size(); i++)
	{
		std::string begin = Game::all_words[i];

		std::vector<int> rounds(Game::all_words.size());
		h_guess_all(Game::all_words, rounds.data(), begin);

		size_t total_guess = 0;
		for (size_t j = 0; j < Game::all_words.size(); j++)
		{
			total_guess += (size_t)rounds[j];
		}
		double ave_guess = (double)(total_guess) / (double)Game::all_words.size();		
		fprintf(fp, "%s, %f\n", begin.c_str(), ave_guess);

		printf("%s, %f   %d/%d\n", begin.c_str(), ave_guess, i, Game::all_words.size());

	}
	fclose(fp);*/


	/*for (int iter = 0; iter<5; iter++)
	{
		h_set_words(Game::all_words);

		std::vector<int> rounds(Game::all_words.size() * Game::all_words.size());
		h_guess_matrix((int)Game::all_words.size(), rounds.data());

		struct Index
		{
			int idx;
			double ave;
		};

		std::vector<Index> index(Game::all_words.size());
		
		for (size_t i = 0; i < Game::all_words.size(); i++)
		{		
			size_t total_guess = 0;
			for (size_t j = 0; j < Game::all_words.size(); j++)
			{
				total_guess += (size_t)rounds[j + i * Game::all_words.size()];
			}
			index[i].idx = i;
			index[i].ave =  (double)(total_guess) / (double)Game::all_words.size();
		}

		
		std::sort(index.begin(), index.end(), [](const Index& a, const Index& b)
		{
			return a.ave < b.ave;
		});

		printf("%s %f\n", Game::all_words[index[0].idx].c_str(), index[0].ave);

		std::vector<std::string> lst_words(Game::all_words.size());
		for (size_t i = 0; i < Game::all_words.size(); i++)
		{
			int j = index[i].idx;
			lst_words[i] = Game::all_words[j];
		}

		Game::all_words = lst_words;

		if (iter == 3)
		{
			FILE* fp = fopen("wordle_machine_optimized.txt", "w");
			for (size_t i = 0; i < Game::all_words.size(); i++)
			{
				std::string word = Game::all_words[i];
				fprintf(fp, "%s\n", word.c_str());

			}
			fclose(fp);
		}
		else if (iter == 4)
		{
			FILE* fp = fopen("op_begins.csv", "w");
			for (size_t i = 0; i < Game::all_words.size(); i++)
			{
				std::string begin = Game::all_words[i];
				double ave_guess = index[i].ave;
				fprintf(fp, "%s, %f\n", begin.c_str(), ave_guess);

			}
			fclose(fp);
		}
	}*/


	return 0;
}

