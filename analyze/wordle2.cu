#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <algorithm>

#define WORD_LENGTH 5
#define NUM_WORDS 2309
#define MASK_LENGTH ((NUM_WORDS+31)>>5)

__constant__ int c_num_words;
__constant__ char c_words[NUM_WORDS][WORD_LENGTH];

inline __device__ void d_judge(int idx_truth, int idx_guess, char feedback[5])
{
	const char* truth = c_words[idx_truth];
	const char* guess = c_words[idx_guess];

	unsigned char mask_used = 0;

	for (int i = 0; i < 5; i++)
	{
		if (guess[i] == truth[i])
		{
			feedback[i] = 2;
			mask_used |= (1 << i);
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
				if ((mask_used & (1 << j)) == 0 && guess[i] == truth[j])
				{
					feedback[i] = 1;
					mask_used |= (1 << j);
					break;
				}
			}
		}
	}
}

inline __device__ int d_new_length(int idx_truth, int idx_guess)
{
	unsigned masks_exclude[WORD_LENGTH];
	for (int i = 0; i < WORD_LENGTH; i++)
	{
		masks_exclude[i] = 0;
	}

	if (idx_guess == idx_truth) return 1;

	char feedback[5];
	d_judge(idx_truth, idx_guess, feedback);

	unsigned char min_counts[26];
	unsigned char max_counts[26];
	for (int i = 0; i < 26; i++)
	{
		min_counts[i] = 0;
		max_counts[i] = 5;
	}

	const char* guess = c_words[idx_guess];

	for (int i = 0; i < 5; i++)
	{
		char c = guess[i];
		int j = feedback[i];

		if (j == 1)
		{
			min_counts[c - 'a']++;
			masks_exclude[i] |= (1 << (c - 'a'));
		}
		else if (j == 2)
		{
			min_counts[c - 'a']++;
			int count_set = __popc(masks_exclude[i]);
			if (count_set < 25)
			{
				for (int k = 0; k < 26; k++)
				{
					char c2 = 'a' + k;
					if (c2 != c)
					{
						masks_exclude[i] |= (1 << (c2 - 'a'));
					}
				}
			}
		}
		else
		{
			masks_exclude[i] |= (1 << (c - 'a'));
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

	int len = 0;
	for (int i = 0; i < c_num_words; i++)
	{
		const char* word = c_words[i];
		bool remove = false;

		unsigned char counts[26];
		for (int j = 0; j < 26; j++)
		{
			counts[j] = 0;
		}

		for (int j = 0; j < 5; j++)
		{
			char c = word[j];
			if ((masks_exclude[j] & (1 << (c - 'a'))) != 0)
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

		if (!remove) len++;
	}

	return len;
}


__global__ void g_calc_lengths(int* lengths)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < NUM_WORDS && j < NUM_WORDS)
	{
		lengths[i + j * NUM_WORDS] = d_new_length(i, j);
	}
}

void h_calc_lengths(const std::vector<std::string>& words, int* lengths)
{
	int num_words = (int)words.size();
	cudaMemcpyToSymbol(c_num_words, &num_words, sizeof(int));

	unsigned char h_words[NUM_WORDS][WORD_LENGTH];
	for (int i = 0; i < num_words; i++)
	{
		memcpy(h_words[i], words[i].c_str(), WORD_LENGTH);
	}
	cudaMemcpyToSymbol(c_words, h_words, num_words * WORD_LENGTH);

	int* d_lengths;
	cudaMalloc(&d_lengths, sizeof(int) * num_words * num_words);

	unsigned num_blocks = (num_words + 7) / 8;
	dim3 blocks = { num_blocks , num_blocks ,1 };
	dim3 block_size = { 8 , 8 ,1 };
	g_calc_lengths << < blocks, block_size >> > (d_lengths);

	cudaMemcpy(lengths, d_lengths, sizeof(int) * num_words * num_words, cudaMemcpyDeviceToHost);

}