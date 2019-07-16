
# Revision control software

J.R. Johansson (jrjohansson at gmail.com) and modified by **Lento Manickathan**

The latest version of this [IPython notebook](http://ipython.org/notebook.html) lecture is available at [http://github.com/jrjohansson/scientific-python-lectures](http://github.com/jrjohansson/scientific-python-lectures).

The other notebooks in this lecture series are indexed at [http://jrjohansson.github.io](http://jrjohansson.github.io).

In any software development, one of the most important tools are revision control software (RCS).

They are used in virtually all software development and in all environments, by everyone and everywhere (no kidding!)

RCS can used on almost any digital content, so it is not only restricted to software development, and is also very useful for manuscript files, figures, data and notebooks!



## There are two main purposes of RCS systems:

1. Keep track of changes in the source code.
    * Allow reverting back to an older revision if something goes wrong.
    * Work on several "branches" of the software concurrently.
    * Tags revisions to keep track of which version of the software that was used for what (for example, "release-1.0", "paper-A-final", ...)
2. Make it possible for serveral people to collaboratively work on the same code base simultaneously.
    * Allow many authors to make changes to the code.
    * Clearly communicating and visualizing changes in the code base to everyone involved.

## Basic principles and terminology for RCS systems

In an RCS, the source code or digital content is stored in a **repository**. 

* The repository does not only contain the latest version of all files, but the complete history of all changes to the files since they were added to the repository. 

* A user can **checkout** the repository, and obtain a local working copy of the files. All changes are made to the files in the local working directory, where files can be added, removed and updated. 

* When a task has been completed, the changes to the local files are **commited** (saved to the repository).

* If someone else has been making changes to the same files, a **conflict** can occur. In many cases conflicts can be **resolved** automatically by the system, but in some cases we might manually have to **merge** different changes together.

* It is often useful to create a new **branch** in a repository, or a **fork** or **clone** of an entire repository, when we doing larger experimental development. The main branch in a repository is called often **master** or **trunk**. When work on a branch or fork is completed, it can be merged in to the master branch/repository.

* With distributed RCSs such as GIT or Mercurial, we can **pull** and **push** changesets between different repositories. For example, between a local copy of there repository to a central online reposistory (for example on a community repository host site like github.com).

### Some good RCS software

1. GIT (`git`) : http://git-scm.com/
2. Mercurial (`hg`) : http://mercurial.selenic.com/

In the rest of this lecture we will look at `git`, although `hg` is just as good and work in almost exactly the same way.

## Installing git

On Linux:
    
    $ sudo apt-get install git

On Mac (with macports):

    $ sudo port install git

The first time you start to use git, you'll need to configure your author information:

    $ git config --global user.name 'Lento Manickathan'
    
    $ git config --global user.email lento.manickathan@gmail.com

## Creating and cloning a repository

To create a brand new empty repository, we can use the command `git init repository-name`:

    $ git init <repository-name>

If we want to fork or clone an existing repository, we can use the command `git clone repository`:

    $ git clone <repository-url>
    

Example: `plotenv` by lento234 at `github`

    $ git clone https://github.com/lento234/plotenv    

We can also clone private repositories over secure protocols such as SSH:

    $ git clone ssh://myserver.com/myrepository

## Status

Using the command `git status` we get a summary of the current status of the working directory. It shows if we have modified, added or removed files.

    $ git status

In this case, only the current ipython notebook has been added. It is listed as an untracked file, and is therefore not in the repository yet.

## Adding files and committing changes

To add a new file to the repository, we first create the file and then use the `git add filename` command:

After having added the file `README`, the command `git status` list it as an *untracked* file.

    $ git add README

    $ git status

Now that it has been added, it is listed as a *new file* that has not yet been commited to the repository.

```
git commit -m "Added a README file" README
```

```
git status 
```

After *committing* the change to the repository from the local working directory, `git status` again reports that working directory is clean.

## Commiting changes

When files that is tracked by GIT are changed, they are listed as *modified* by `git status`:

    $ git status

Again, we can commit such changes to the repository using the `git commit -m "message"` command. and change the `README` file

    $ git commit -m "added one more line in README" README

## Commit logs

The messages that are added to the commit command are supposed to give a short (often one-line) description of the changes/additions/deletions in the commit. If the `-m "message"` is omitted when invoking the `git commit` message an editor will be opened for you to type a commit message (for example useful when a longer commit message is requried). 

We can look at the revision log by using the command `git log`:

    $ git log

In the commit log, each revision is shown with a timestampe, a unique has tag that, and author information and the commit message.

## Diffs

All commits results in a changeset, which has a "diff" describing the changes to the file associated with it. We can use `git diff` so see what has changed in a file:

```
git diff README
```

That looks quite cryptic but is a standard form for describing changes in files. We can use other tools, like graphical user interfaces or web based systems to get a more easily understandable diff.

In github (a web-based GIT repository hosting service) it can look like this:

## Discard changes in the working directory

To discard a change (revert to the latest version in the repository) we can use the `checkout` command like this:

    $ git checkout -- README
    
    $ git status

## Checking out old revisions

If we want to get the code for a specific revision, we can use "git checkout" and giving it the hash code for the revision we are interested as argument:

    $ git log
    
    $ git checkout <hash-code>

Now the content of all the files like in the revision with the hash code listed above (first revision)

We can move back to "the latest" (master) with the command:

    $ git checkout master 

## Tagging and branching

### Tags

Tags are named revisions. They are useful for marking particular revisions for later references. For example, we can tag our code with the tag "paper-1-final" when when simulations for "paper-1" are finished and the paper submitted. Then we can always retreive the exactly the code used for that paper even if we continue to work on and develop the code for future projects and papers.

    $ git tag -a demotag1 -m "Code used for this and that purpuse" 

    $ git tag -l 
    
    $ git show demotag1

To retreive the code in the state corresponding to a particular tag, we can use the `git checkout tagname` command:

    $ git checkout demotag1

## Branches

With branches we can create diverging code bases in the same repository. They are for example useful for experimental development that requires a lot of code changes that could break the functionality in the master branch. Once the development of a branch has reached a stable state it can always be merged back into the trunk. Branching-development-merging is a good development strategy when serveral people are involved in working on the same code base. But even in single author repositories it can often be useful to always keep the master branch in a working state, and always branch/fork before implementing a new feature, and later merge it back into the main trunk.

In GIT, we can create a new branch like this:

    $ git branch <branch-name> 

We can list the existing branches like this:

    $ git branch

And we can switch between branches using `checkout`:

    $ git checkout <branch-name>

Switch to master    

    $ git checkout master

We can merge an existing branch and all its changesets into another branch (for example the master branch) like this:

First change to the target branch:

    $ git merge <branch-name>

We can delete the branch `expr1` now that it has been merged into the master:

    $ git branch -d <branch-name>

## pulling and pushing changesets between repositories

If the respository has been cloned from another repository, for example on github.com, it automatically remembers the address of the parant repository (called origin):

    $ git remote

    $ git remote show origin

### pull

We can retrieve updates from the origin repository by "pulling" changesets from "origin" to our repository:

    $ git pull origin

We can register addresses to many different repositories, and pull in different changesets from different sources, but the default source is the origin from where the repository was first cloned (and the work origin could have been omitted from the line above).

### push

After making changes to our local repository, we can push changes to a remote repository using `git push`. Again, the default target repository is `origin`, so we can do:

    $ git push

## Hosted repositories

Github.com is a git repository hosting site that is very popular with both open source projects (for which it is free) and private repositories (for which a subscription might be needed).

With a hosted repository it easy to collaborate with colleagues on the same code base, and you get a graphical user interface where you can browse the code and look at commit logs, track issues etc. 

Some good hosted repositories are

* Github : http://www.github.com
* Bitbucket: http://www.bitbucket.org

## Further reading

* http://git-scm.com/book
* http://www.vogella.com/articles/Git/article.html
* http://cheat.errtheblog.com/s/git
