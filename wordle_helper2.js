(async () =>
{
    let response = await fetch("wordle_machine_optimized.txt");
    const word_list = await response.text(); 
    let words = word_list.split(/\r?\n/);
    
    const main_content = document.getElementById("main_content");
    
    const container = document.createElement("div");
    main_content.append(container);
       
    {
        const div_list = document.createElement("div");
        div_list.style.cssText = "display: block; float: left; text-align: center;";
        container.append(div_list);
        
        const label_list = document.createElement("div");
        label_list.innerHTML="<p>Options</p>"
        div_list.append(label_list);
        
        div_word_list = document.createElement("div");
        div_word_list.style.cssText = "overflow-y: scroll; height:250px; width:100px;";        
        
        div_list.append(div_word_list);
    }
    
    const update_list = () =>
    {
        div_word_list.innerHTML ="";
        for (let word of options)
        {
            const div_word = document.createElement("div");
            div_word.style.cssText = "user-select: none;";
            div_word.innerHTML = word;
            
            div_word.addEventListener('mouseover',() => {
               div_word.style.backgroundColor ="skyblue";
            });
            
            div_word.addEventListener('mouseleave',() =>{
               div_word.style.backgroundColor = null;
            });
            
            div_word.addEventListener('click', () =>{
                guess_input.value = word;
            });
            
            div_word_list.append(div_word);
        }
        guess_input.value = options[0];
    }
    
    const create_counts = (value = 0) =>
    {
        let counts = {};
        for (let cc = 65; cc< 91; cc++)
        {
            let c = String.fromCharCode(cc);
            counts[c] = value;
        }
        return counts;
    }
    
    const reset = () =>
    {
        options = [...words];
        update_list();
        guess_input.value = "slate";
        feedback_input.value = "";
        div_status_content.innerHTML ="";        
        
        min_counts = create_counts();
        max_counts = create_counts(5);
        exclude_sets = [new Set(), new Set(), new Set(), new Set(), new Set()]
    }
    
    const filter = () =>
    {
        const guess = guess_input.value.toUpperCase();
        if (guess.length!=5) return;
        
        const feedback = feedback_input.value;
        if (feedback.length!=5) return;
        
        const div_line = document.createElement("div");
        div_status_content.append(div_line);
        
        let min_counts1 = create_counts();
        let max_counts1 = create_counts(5);
        
        for (let i =0; i<5; i++)
        {
            let c = guess[i];
            let j = feedback[i];
            
            const letter = document.createElement("div");
            letter.style.cssText = "float: left; color: white; width:40px; height:40px; font-size: 30px;";            
            letter.innerHTML = c;
            if (j==1)
            {
                letter.style.backgroundColor = "#c9b458";
            }
            else if (j==2)
            {
                letter.style.backgroundColor = "#6aaa64";
            }
            else
            {
                letter.style.backgroundColor = "#787c7e";
            }
            div_line.append(letter);
        }
        
        for (let i =0; i<5; i++)
        {
            let c = guess[i];
            let j = feedback[i];
            
            if (j==1)
            {
                min_counts1[c]++;
                exclude_sets[i].add(c);
            }
            else if (j==2)
            {
                min_counts1[c]++;
                if (exclude_sets[i].size<25)
                {
                    for (let cc = 65; cc< 91; cc++)
                    {
                        let c2 = String.fromCharCode(cc);
                        if (c2!=c) 
                        {
                            exclude_sets[i].add(c2);
                        }
                    }
                }
            }
            else
            {
                exclude_sets[i].add(c);
            }
        }
        
        for (let i =0; i<5; i++)
        {
            let c = guess[i];
            let j = feedback[i];
            if (j!=1 && j!=2)
            {
                max_counts1[c] = min_counts1[c];
            }          
        }
        
        for (let cc = 65; cc< 91; cc++)
        {
            let c = String.fromCharCode(cc);
            if (min_counts1[c] > min_counts[c])
            {
                min_counts[c] = min_counts1[c];
            }
            if (max_counts1[c] < max_counts[c])
            {
                max_counts[c] = max_counts1[c];
            }
        }
        
        for (let i = 0; i< options.length; i++)
        {
            const uword = options[i].toUpperCase();
            let remove = false;
            let counts = create_counts();            
            for (let j=0;j<5;j++)
            {
                let c = uword[j];
                if (exclude_sets[j].has(c))
                {
                    remove = true;
                    break;
                }
                counts[c]++;
            }
            if (!remove)
            {
                for(let c in counts)
                {
                    let min_count = min_counts[c];
                    let max_count = max_counts[c];
                    let count = counts[c];
                    if (count<min_count || count>max_count)
                    {
                        remove = true;
                        break;
                    }
                }
            }
            
            if (remove)
            {
                options.splice(i, 1);
                i--;
            }
        }
        
        update_list();
        feedback_input.value = "";
    }
    
    {
        const div_guess = document.createElement("div");
        div_guess.style.cssText = "display: block; float: left; text-align: center;";
        container.append(div_guess);
        
        const label_guess = document.createElement("div");
        label_guess.innerHTML="<p>Guess</p>"
        div_guess.append(label_guess);
        
        guess_input = document.createElement("input");
        guess_input.type = "text";
        guess_input.style.cssText = "text-align: center;";
        div_guess.append(guess_input);
        
        const label_feedback = document.createElement("div");
        label_feedback.innerHTML="<p>Feedback</p><p>0: grey; 1: yellow; 2: green;</p>"
        div_guess.append(label_feedback);
        
        feedback_input = document.createElement("input");
        feedback_input.type = "text";
        feedback_input.style.cssText = "text-align: center;";
        div_guess.append(feedback_input);
        
        let br = document.createElement("br");
        div_guess.append(br);
        
        const btn_filter = document.createElement("button");
        btn_filter.innerHTML = "Filter";
        btn_filter.addEventListener('click', filter);
        div_guess.append(btn_filter);
        
        br = document.createElement("br");
        div_guess.append(br);
        
        const btn_reset = document.createElement("button");
        btn_reset.innerHTML = "Reset";
        btn_reset.addEventListener('click', reset);
        div_guess.append(btn_reset);
    }
    
    {
        const div_status = document.createElement("div");
        div_status.style.cssText = "display: block; float: left; text-align: center;";
        container.append(div_status);
        
        const label_status = document.createElement("div");
        label_status.innerHTML="<p>Status</p>"
        div_status.append(label_status);
        
        div_status_content = document.createElement("div");
        div_status_content.style.cssText = "width:200px; height:300px";
        div_status.append(div_status_content);
        
    }
    
    reset();

})();
