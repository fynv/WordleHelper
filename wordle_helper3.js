(async () =>
{
    const response = await fetch("wordle_machine_optimized.txt");
    const word_list = await response.text(); 
    const words = word_list.split(/\r?\n/);
    
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
    }
    
    const judge = (truth, guess) =>
    {
        feedback = [0,0,0,0,0];
        used = [0,0,0,0,0];        
        
        for (let i = 0; i < 5; i++)
        {
            if (guess[i] == truth[i])
            {
                feedback[i] = 2;
                used[i] = 1;
            }
        }

        for (let i = 0; i < 5; i++)
        {
            if (feedback[i] == 0)
            {
                for (let j = 0; j < 5; j++)
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

        return feedback;
    }
    
    const read_opt = async (fn_opt) =>
    {
        const response = await fetch("data1/"+fn_opt);
        const opt = await response.text(); 
        const lines = opt.split(/\r?\n/);
        opt_suggestion = lines[0];
        opt_dict = {};
        
        for (let i=1; i<lines.length; i++)
        {
            let line = lines[i];
            let item = line.split(' ');
            opt_dict[item[0]] = item[1];
        }
        guess_input.value = opt_suggestion;
    }
    
    const clean_opt = ()=>
    {
        opt_suggestion = "#####";
        opt_dict = {};
        guess_input.value = options[0];
    }
    
    const reset = () =>
    {
        options = [...words];
        update_list();
        
        read_opt("d64d8a103311a203");
        
        feedback_input.value = "";
        div_status_content.innerHTML ="";        
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
        
        for (let i = 0; i< options.length; i++)
        {
            const uword = options[i].toUpperCase();
            let remove = false;
            
            let feedback2 = judge(uword, guess);
            for (let j = 0; j < 5; j++)
            {
                if (feedback2[j] != feedback[j])
                {
                    remove = true;
                    break;
                }
            }
          
            if (remove)
            {
                options.splice(i, 1);
                i--;
            }
        }
        
        update_list();
        
        let has_opt = false;
        if (opt_suggestion != "#####" && guess == opt_suggestion.toUpperCase())
        {
            if (opt_dict.hasOwnProperty(feedback))
            {
                let fn_opt = opt_dict[feedback];
                if (fn_opt!='0')
                {
                    read_opt(fn_opt);
                    has_opt = true;
                }
            }
        }
        
        if (!has_opt)
        {
            clean_opt();            
        }
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
