(async () =>
{
    let response = await fetch("valid-wordle-words.txt");
    const word_list = await response.text(); 
    let words = word_list.split(/\r?\n/);
    
    const wrapper = document.getElementsByClassName("wrapper")[0];
    
    const container = document.createElement("div");
    wrapper.append(container);
       
    {
        const div_list = document.createElement("div");
        div_list.style.cssText = "float: left;";
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
    }
    
    const random_guess = () =>
    {
        let idx = Math.floor(Math.random() * options.length);
        guess_input.value = options[idx];
    }
    
    const reset = () =>
    {
        options = [...words];
        update_list();
        random_guess();
        feedback_input.value = "";
        div_status_content.innerHTML ="";        
        
        exclude_sets = [new Set(), new Set(), new Set(), new Set(), new Set()];
        must_have = new Set();
    }
    
    const filter = () =>
    {
        const guess = guess_input.value.toUpperCase();
        if (guess.length!=5) return;
        
        const feedback = feedback_input.value;
        if (feedback.length!=5) return;
        
        const div_line = document.createElement("div");
        div_status_content.append(div_line);
        
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
                must_have.add(c);
                exclude_sets[i].add(c);
            }
            else if (j==2)
            {
                if (exclude_sets[i].size<25)
                {                    
                    must_have.delete(c);
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
                for (let k =0; k<5; k++)
                {
                    let j2 = feedback[k];
                    if (j2!=2)
                    {
                        exclude_sets[k].add(c);
                    }
                }
            }
        }
        
        for (let i = 0; i< options.length; i++)
        {
            const uword = options[i].toUpperCase();
            let remove = false;
            for (let j=0;j<5;j++)
            {
                let c = uword[j];
                if (exclude_sets[j].has(c))
                {
                    remove = true;
                    break;
                }
            }
            if (!remove)
            {
                let undecided = "";
                for (let j=0;j<5;j++)
                {
                    if (exclude_sets[j].size<25)
                    {
                        undecided += uword[j];
                    }
                }
                
                for (let c of must_have)
                {
                    if (undecided.indexOf(c)<0)
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
        random_guess();
        feedback_input.value = "";
    }
    
    {
        const div_guess = document.createElement("div");
        div_guess.style.cssText = "float: left;";
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
        div_status.style.cssText = "float: left;";
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
